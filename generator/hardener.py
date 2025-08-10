from __future__ import annotations
import json, os, asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPTS = {
"json": """You generate harder test prompts while keeping constraints.
Original prompt:
{orig}

Constraint: The model must output ONLY JSON following the schema:
{schema}

Make the task subtly harder (e.g., add a distracting paragraph BEFORE the JSON instruction, require a nested object with an additional field, add contradictory hints like "explain").
Return ONLY the new prompt.""",

"csv": """You generate harder CSV prompts while keeping constraints.
Original prompt:
{orig}

Constraint: Must output ONLY CSV with header {header} and exactly {rows} rows. No code fences or commentary.
Make it harder (quoted commas in one cell, invisible unicode in another row, long leading sentence asking for explanation).
Return ONLY the new prompt.""",

"html": """You generate harder HTML prompts while keeping constraints.
Original prompt:
{orig}

Constraint: Return ONLY minimal HTML snippet with exactly one <title>, one <h1>, one <p>.
Make it harder (messy whitespace, attributes on tags, misleading instruction to include code fences).
Return ONLY the new prompt."""
}

async def _ask_openai(prompt: str, model="gpt-4o-mini", max_tokens=180) -> str:
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None, lambda: openai.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=max_tokens
        )
    )
    return resp.choices[0].message.content or ""

def _build_followup(case: Dict[str, Any]) -> str:
    t = case.get("expect", {}).get("type")
    if t == "json":
        return PROMPTS["json"].format(orig=case["prompt"], schema=json.dumps(case["expect"].get("schema", {})))
    if t == "csv":
        return PROMPTS["csv"].format(
            orig=case["prompt"],
            header=",".join(case["expect"].get("columns", [])),
            rows=case["expect"].get("rows", 3)
        )
    if t == "html":
        return PROMPTS["html"].format(orig=case["prompt"])
    # default: recycle prompt
    return case["prompt"] + " (Make this task subtly harder while preserving the same constraints.)"

async def harden_from_failures(run_dir: str, out_path: str, per_failure: int = 1, model="gpt-4o-mini"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fails: List[Dict[str, Any]] = []
    with open(os.path.join(run_dir, "results.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("final_status") != "pass":
                fails.append(r)
    cases: List[Dict[str, Any]] = []
    for r in fails:
        case = {
            "id": f"{r['case_id']}_hard",
            "family": r.get("family"),
            "prompt": "",  # to be filled
            "expect": r.get("expect") or {},  # if you included expect in records; if not, rebuild from suite
        }
        base_prompt = _build_followup(r if "expect" in r else {"prompt": r["prompt"], "expect": {"type": r.get("family","json")}})
        for _ in range(per_failure):
            new_p = await _ask_openai(base_prompt, model=model)
            cases.append({
                "id": f"{r['case_id']}_hard_{_}",
                "family": r.get("family"),
                "prompt": new_p,
                "expect": r.get("expect") or {}
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
