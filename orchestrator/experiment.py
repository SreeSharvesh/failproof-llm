from __future__ import annotations
import yaml
from typing import Any, Dict

def load_experiment(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
