from __future__ import annotations
import asyncio, json, re, time, uuid
from typing import Any, Dict, List, Tuple
import functools
from orchestrator.storage import Storage
from validators.schema_validator import validate_json
from validators.csv_validator import validate_csv
from validators.html_validator import validate_html
from strategies.auto_repair import extract_json
from strategies.auto_repair_structured import repair_csv, repair_html
from validators.format_explainer import get_format_explanation
from validators.domain.financial_validator import validate_fin_invoice_json
from validators.domain.healthcare_validator import validate_hc_claim_csv
from validators.domain.legal_validator import validate_legal_html_snippet
from configs.expect_registry import expect_for_family


# ---------- helpers ----------
async def _call_model(adapter, prompt: str, **overrides) -> Any:
    loop = asyncio.get_running_loop()
    fn = functools.partial(adapter.generate, prompt, **overrides)
    return await loop.run_in_executor(None, fn)

def _parse_retry_delay_seconds(error_msg: str) -> int:
    if not error_msg: return 60
    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_msg, re.I)
    if m:
        try: return max(1, int(m.group(1)))
        except: pass
    m2 = re.search(r"retry[-_ ]?after[:\s]+(\d+)", error_msg, re.I)
    if m2:
        try: return max(1, int(m2.group(1)))
        except: pass
    return 60

def _classify_error(error: str | None) -> Tuple[str | None, str]:
    if not error: return None, "unknown"
    e = error.lower()
    if "timeout" in e or "timed out" in e: return "timeout", "fail"
    if "429" in e or "rate limit" in e or "quota" in e: return "rate_limited", "fail"
    return "crash", "fail"

def validate_by_case(output: str, case: dict) -> tuple[bool, str, dict]:
    validators = {}
    expect = case.get("expect", {})
    t = expect.get("type")

    if not output or not output.strip():
        return False, "empty_output", validators

    # core format validators (you already have these)
    if t == "json":
        ok, reason = validate_json(output, expect.get("schema", {}))
        validators["json_schema"] = ok
    elif t == "csv":
        ok, reason = validate_csv(output, expect)
        validators["csv"] = ok
    elif t == "html":
        ok, reason = validate_html(output, expect)
        validators["html"] = ok
    else:
        ok, reason = True, "no_validation"

    # domain-specific augment (only if core ok)
    if ok:
        fam = (case.get("family") or "")
        if fam.startswith("fin_") and t == "json":
            ok, reason = validate_fin_invoice_json(output)
            validators["domain_finance"] = ok
        elif fam.startswith("hc_") and t == "csv":
            ok, reason = validate_hc_claim_csv(output)
            validators["domain_healthcare"] = ok
        elif fam.startswith("lg_") and t == "html":
            ok, reason = validate_legal_html_snippet(output)
            validators["domain_legal"] = ok

    return ok, reason, validators

def _reason_to_taxonomy(reason: str) -> str:
    r = (reason or "").lower()
    if not r or r in ("ok","no_validation"): return ""
    structural = ("parse_error","schema_error","bad_header","wrong_row_count",
                  "column_mismatch","missing","multiple","extraneous","invalid")
    return "invalid_structure" if any(k in r for k in structural) else "invalid_output"


# ---------- main ----------
async def run_experiment(
    cases: List[Dict[str, Any]],
    adapters: Dict[str, Any],
    run_dir: str,
    cfg: Dict[str, Any],
):
    storage = Storage(run_dir)
    run_params = cfg.get("run", {})
    guards = cfg.get("guards", {}) or {}
    critics = cfg.get("critics", {}) or {}

    parallelism = int(run_params.get("parallelism", 8))
    timeout_s = int(run_params.get("timeout_s", 20))
    retries = int(run_params.get("retries", 1))
    sc_json_n = int(guards.get("self_consistency_json", 1))
    enable_repairs = bool(guards.get("repairs", True))

    sem = asyncio.Semaphore(parallelism)

    async def worker(case: Dict[str, Any], model_key: str):
        adapter = adapters[model_key]
        prompt = case["prompt"]
        attempt = 0
        result_obj = None
    
        # (optional but recommended) if you added the canonical expect registry:
        from configs.expect_registry import expect_for_family
        fam = case.get("family") or ""
        canon = expect_for_family(fam)
        if canon:
            case["expect"] = canon
    
        expect = case.get("expect", {}) or {}
        expects_json = (expect.get("type") == "json")
    
        async with sem:
            # ---- self-consistency for JSON family (optional) ----
            if expects_json and sc_json_n > 1:
                texts, errs = [], []
                for _ in range(sc_json_n):
                    attempt += 1
                    start_wall = time.time()
                    try:
                        res_n = await asyncio.wait_for(
                            _call_model(adapter, prompt, force_json=expects_json),
                            timeout=timeout_s
                        )
                    except asyncio.TimeoutError:
                        res_n = type("Result", (), {})()
                        res_n.text, res_n.usage, res_n.error = "", {}, "timeout"
                        res_n.latency_ms = int((time.time() - start_wall) * 1000)
    
                    # simple 429 retry
                    if res_n.error and ("429" in str(res_n.error) or "rate limit" in str(res_n.error).lower() or "quota" in str(res_n.error).lower()):
                        if attempt <= retries:
                            await asyncio.sleep(min(_parse_retry_delay_seconds(str(res_n.error)), 5))
                            continue
    
                    if res_n.text:
                        texts.append(res_n.text)
                    if res_n.error:
                        errs.append(str(res_n.error))
    
                # choose first valid by validator; else majority string
                chosen_text = ""
                for t in texts:
                    ok, _, _ = validate_by_case(t, case)
                    if ok:
                        chosen_text = t
                        break
                if not chosen_text and texts:
                    from collections import Counter
                    chosen_text = Counter([t.strip() for t in texts]).most_common(1)[0][0]
    
                result_obj = type("Result", (), {})()
                result_obj.text = chosen_text
                result_obj.usage = {}
                result_obj.latency_ms = 0
                result_obj.error = errs[0] if (not chosen_text and errs) else None
    
            # ---- single call path ----
            if result_obj is None:
                while True:
                    attempt += 1
                    start_wall = time.time()
                    try:
                        res = await asyncio.wait_for(
                            _call_model(adapter, prompt, force_json=expects_json),  # <-- pass expects_json here
                            timeout=timeout_s
                        )
                    except asyncio.TimeoutError:
                        res = type("Result", (), {})()
                        res.text, res.usage, res.error = "", {}, "timeout"
                        res.latency_ms = int((time.time() - start_wall) * 1000)
    
                    if res.error and ("429" in str(res.error) or "rate limit" in str(res.error).lower() or "quota" in str(res.error).lower()):
                        if attempt <= retries:
                            await asyncio.sleep(min(_parse_retry_delay_seconds(str(res.error)), 5))
                            continue
                    result_obj = res
                    break
    
        # ---- base record (unchanged) ----
        record: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "case_id": case.get("id"),
            "family": case.get("family"),
            "model": model_key,
            "prompt": prompt[:2000],
            "output": getattr(result_obj, "text", "") or "",
            "latency_ms": getattr(result_obj, "latency_ms", 0),
            "error": getattr(result_obj, "error", None),
            "validators": {},
            "taxonomy": None,
            "strategies_applied": [],
            "final_status": "unknown",
            "status_before": "unknown",
            "taxonomy_before": None,
            "explanations": {},
        }

        # Provider/infra error path
        if record["error"]:
            tax, status = _classify_error(record["error"])
            record["taxonomy_before"] = tax
            record["status_before"] = "fail"
            record["taxonomy"] = tax
            record["final_status"] = status
            storage.log(record)
            return

        # Validate first (BEFORE repairs)
        ok0, reason0, vmap0 = validate_by_case(record["output"], case)
        record["status_before"] = "pass" if ok0 else "fail"
        record["taxonomy_before"] = None if ok0 else _reason_to_taxonomy(reason0)

        if ok0 or not enable_repairs:
            # no repairs applied
            record["validators"] = vmap0
            record["taxonomy"] = None if ok0 else record["taxonomy_before"]
            record["final_status"] = "pass" if ok0 else "fail"
            # AI explainer only for fails
            if (not ok0) and (critics.get("format_explainer", {}).get("enabled")):
                expl = await get_format_explanation(critics["format_explainer"], case, record["output"], reason0)
                if expl: record["explanations"]["format"] = expl
            storage.log(record)
            return

        # Try repairs (JSON/CSV/HTML)
        strategies_used: List[str] = []
        final_output = record["output"]
        expect = case.get("expect", {})
        t = expect.get("type")

        ok, reason, vmap = ok0, reason0, vmap0  # start with initial

        if not ok:
            if t == "json":
                extracted = extract_json(record["output"])
                if extracted:
                    ok2, reason2, vmap2 = validate_by_case(extracted, case)
                    if ok2:
                        final_output, ok, vmap = extracted, True, vmap2
                        strategies_used.append("auto_repair_extract_json")
                    else:
                        reason = reason2  # keep latest reason
            elif t == "csv":
                fixed = repair_csv(record["output"], expect.get("columns", []), expect.get("rows", 0))
                if fixed:
                    ok2, reason2, vmap2 = validate_by_case(fixed, case)
                    if ok2:
                        final_output, ok, vmap = fixed, True, vmap2
                        strategies_used.append("auto_repair_repair_csv")
                    else:
                        reason = reason2
            elif t == "html":
                fixed = repair_html(record["output"])
                if fixed:
                    ok2, reason2, vmap2 = validate_by_case(fixed, case)
                    if ok2:
                        final_output, ok, vmap = fixed, True, vmap2
                        strategies_used.append("auto_repair_repair_html")
                    else:
                        reason = reason2

        # finalize after repairs
        record["output"] = final_output
        record["validators"] = vmap
        record["strategies_applied"] = strategies_used
        record["final_status"] = "pass" if ok else "fail"
        record["taxonomy"] = None if ok else _reason_to_taxonomy(reason)

        # AI explainer for remaining fails
        if (not ok) and (critics.get("format_explainer", {}).get("enabled")):
            expl = await get_format_explanation(critics["format_explainer"], case, record["output"], reason)
            if expl: record["explanations"]["format"] = expl

        storage.log(record)

    tasks: List[asyncio.Task] = []
    for case in cases:
        for model_key in adapters.keys():
            tasks.append(asyncio.create_task(worker(case, model_key)))
    await asyncio.gather(*tasks)
