from __future__ import annotations
import json, collections, statistics, os
from typing import Dict, Any

def summarize(run_path: str) -> dict:
    fam_counts = collections.Counter()
    fam_pass = collections.Counter()
    tax_counts = collections.Counter()
    strat_fixed = collections.Counter()
    latencies = []
    repairs_used = collections.Counter()
    flips = 0

    results_path = os.path.join(run_path, "results.jsonl")
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            fam = r.get("family","unknown")
            fam_counts[fam] += 1
            if r.get("final_status") == "pass":
                fam_pass[fam] += 1
            tax = r.get("taxonomy")
            if tax: tax_counts[tax] += 1
            if r.get("latency_ms") is not None:
                latencies.append(r["latency_ms"])

            # strategy effectiveness
            sb = r.get("status_before")
            sa = r.get("final_status")
            if sb == "fail" and sa == "pass":
                flips += 1
                for s in r.get("strategies_applied", []):
                    strat_fixed[s] += 1
            for s in r.get("strategies_applied", []):
                repairs_used[s] += 1

    by_family = {
        fam: {
            "count": fam_counts[fam],
            "pass": fam_pass[fam],
            "pass_rate": (fam_pass[fam]/fam_counts[fam]) if fam_counts[fam] else 0.0
        } for fam in fam_counts
    }
    lat = {
        "p50": statistics.median(latencies) if latencies else None,
        "p95": (sorted(latencies)[int(0.95*len(latencies))-1] if latencies else None)
    }
    return {
        "by_family": by_family,
        "taxonomy": dict(tax_counts),
        "latency": lat,
        "strategy_effectiveness": {
            "flips_fail_to_pass": flips,
            "fixed_by_strategy": dict(strat_fixed),
            "repairs_used_counts": dict(repairs_used)
        }
    }
