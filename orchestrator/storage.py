from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any

class Storage:
    def __init__(self, run_dir: str):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.results_path = os.path.join(run_dir, "results.jsonl")

    def log(self, record: Dict[str, Any]):
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
