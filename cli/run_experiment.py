from __future__ import annotations
import os, sys, json, asyncio, argparse
from dotenv import load_dotenv
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.experiment import load_experiment
from orchestrator.runner import run_experiment
from adapters.openai_adapter import OpenAIAdapter
from adapters.gemini_adapter import GeminiAdapter

def load_cases(path: str) -> List[Dict[str, Any]]:
    # Expect JSONL: {"id": "...", "family": "json_schema", "prompt": "..."}
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

# cli/run_experiment.py (build_adapters)
def build_adapters(cfg):
    adapters = {}
    for m in cfg["models"]:
        key = m["key"]
        params = m.get("params", {})
        if m["provider"] == "openai":
            adapters[key] = OpenAIAdapter(m["model"], **params)   # includes default force_json if present
        # elif m["provider"] == "gemini":
        #     adapters[key] = GeminiAdapter(m["model"], **params)
        else:
            raise ValueError(f"Unknown provider: {m['provider']}")
    return adapters


if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    cfg = load_experiment(args.experiment)
    cases = load_cases(cfg["suite"])
    adapters = build_adapters(cfg)
    asyncio.run(run_experiment(cases, adapters, args.run_dir, cfg.get("run", {})))
    print(f"Run complete → {args.run_dir}")
