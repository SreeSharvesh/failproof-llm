from __future__ import annotations
import os, json, math, pathlib, uuid, time
from typing import Dict, Any, List, Tuple
from string import Template

# Uses OpenAI; set OPENAI_API_KEY in env
import openai

# ---------- Template discovery ----------

def discover_templates(root: str = "templates") -> List[Dict[str, str]]:
    """
    Recursively scan templates/ for .md files.
    Returns a list of {key, label, path, group} where:
      - key: file stem (e.g., struct_json_malformed)
      - label: pretty name (Group / Humanized)
      - path: absolute path to .md
      - group: top-level folder (structural/adversarial/...)
    """
    results: List[Dict[str, str]] = []
    base = pathlib.Path(root)
    if not base.exists():
        return results
    for p in base.rglob("*.md"):
        group = p.parent.parent.name if p.parent.parent != base else p.parent.name
        key = p.stem  # e.g., struct_json_malformed
        label = f"{group.title()} / " + key.replace("_", " ").title()
        results.append({"key": key, "label": label, "path": str(p.resolve()), "group": group})
    # stable sort by group then key
    results.sort(key=lambda x: (x["group"], x["key"]))
    return results

# ---------- Prompt rendering ----------

UNIVERSAL_FOOTER_TMPL = Template("""
Return a JSON array of exactly ${NUM_ITEMS} objects with fields:
- id (string), family (string), category (string), prompt (string), expect (object), meta (object).

Contracts for validators:
- JSON tasks: expect = { "type":"json", "schema": SCHEMA_OBJECT }
- CSV  tasks: expect = { "type":"csv", "columns":[...], "rows":3, "strict_no_extra_text":true }
- HTML tasks: expect = { "type":"html", "required_tags":[...], "strict_no_extra_text":true }
- TEXT tasks: expect = { "type":"text", "policy":[...optional policy keys...] }

Additional constraints:
- IDs must be unique within this batch (e.g., "${FAMILY}_0001", ...).
- Always set family="${FAMILY}" and category="${CATEGORY_LABEL}".
- meta must include: {"difficulty":"base","mutations":[], "seed": ${SEED}}
Output only the JSON array. No code fences. No extra text.
""")

def _read(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")

def _render_full_prompt(template_path: str, *, family: str, category_label: str, num_items: int, seed: int) -> str:
    body = _read(template_path)
    footer = UNIVERSAL_FOOTER_TMPL.safe_substitute(
        NUM_ITEMS=num_items, FAMILY=family, CATEGORY_LABEL=category_label, SEED=seed
    )
    # Support ${VAR} interpolation inside the template (optional)
    body = Template(body).safe_substitute(NUM_ITEMS=num_items, FAMILY=family, CATEGORY_LABEL=category_label, SEED=seed)
    return body.strip() + "\n\n" + footer.strip()

# ---------- OpenAI call ----------

def _call_openai(prompt: str, *, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 4000) -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content or ""

# ---------- Output handling ----------

def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Top-level is not an array")
        return data
    except Exception as e:
        raise ValueError(f"Model did not return a valid JSON array: {e}\n\n{text[:400]}")

def _normalize_records(arr: List[Dict[str, Any]], family: str, category_label: str, seed: int) -> List[Dict[str, Any]]:
    out = []
    for i, obj in enumerate(arr):
        obj = dict(obj)
        obj.setdefault("id", f"{family}_{i:04d}")
        obj.setdefault("family", family)
        obj.setdefault("category", category_label)
        meta = obj.get("meta") or {}
        meta.setdefault("difficulty", "base")
        meta.setdefault("mutations", [])
        meta.setdefault("seed", seed)
        obj["meta"] = meta
        out.append(obj)
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Public API ----------

def generate_from_template(
    *, template_path: str, out_path: str, family: str, category_label: str,
    num_items: int, seed: int = 1234, model: str = "gpt-4o-mini", temperature: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Single-shot generation (up to ~100 items comfortably). Returns the records and writes JSONL.
    """
    prompt = _render_full_prompt(template_path, family=family, category_label=category_label, num_items=num_items, seed=seed)
    text = _call_openai(prompt, model=model, temperature=temperature)
    arr = _parse_json_array(text)
    rows = _normalize_records(arr, family, category_label, seed)
    write_jsonl(out_path, rows)
    return rows

def generate_from_template_chunked(
    *, template_path: str, out_path: str, family: str, category_label: str,
    total_items: int, chunk_size: int = 50, seed: int = 1234, model: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> List[Dict[str, Any]]:
    """
    For large datasets: split into batches so the model returns consistent JSON.
    """
    all_rows: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        chunks = math.ceil(total_items / chunk_size)
        for c in range(chunks):
            n = chunk_size if (c < chunks - 1) else (total_items - chunk_size * (chunks - 1))
            # vary seed slightly per chunk for diversity, but keep logged
            chunk_seed = seed + c
            prompt = _render_full_prompt(template_path, family=family, category_label=category_label, num_items=n, seed=chunk_seed)
            text = _call_openai(prompt, model=model, temperature=temperature)
            arr = _parse_json_array(text)
            rows = _normalize_records(arr, family, category_label, chunk_seed)

            # Re-ID to keep global uniqueness
            for i, r in enumerate(rows):
                r["id"] = f"{family}_{c:02d}_{i:04d}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

            all_rows.extend(rows)
    return all_rows
