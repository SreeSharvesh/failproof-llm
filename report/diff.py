from __future__ import annotations
import json, os
from report.metrics import summarize

def diff_runs(off_dir: str, on_dir: str) -> dict:
    off = summarize(off_dir)
    on  = summarize(on_dir)

    fams = sorted(set(off["by_family"].keys()) | set(on["by_family"].keys()))
    by_family = {}
    for fam in fams:
        off_p = off["by_family"].get(fam, {}).get("pass_rate", 0.0)
        on_p  = on["by_family"].get(fam, {}).get("pass_rate", 0.0)
        by_family[fam] = {"off": off_p, "on": on_p, "delta": on_p - off_p}

    def _cnt(d): return sum(d.values()) if d else 0
    tax_shift = {
        "off_total_failures": _cnt(off.get("taxonomy", {})),
        "on_total_failures": _cnt(on.get("taxonomy", {})),
        "delta_failures": _cnt(on.get("taxonomy", {})) - _cnt(off.get("taxonomy", {}))
    }

    lat = {
        "off_p50": off["latency"]["p50"], "on_p50": on["latency"]["p50"],
        "off_p95": off["latency"]["p95"], "on_p95": on["latency"]["p95"],
        "delta_p50": (on["latency"]["p50"] or 0) - (off["latency"]["p50"] or 0),
        "delta_p95": (on["latency"]["p95"] or 0) - (off["latency"]["p95"] or 0),
    }

    return {
        "by_family": by_family,
        "taxonomy_shift": tax_shift,
        "latency_shift": lat,
        "on_strategy_effectiveness": on.get("strategy_effectiveness", {})
    }
