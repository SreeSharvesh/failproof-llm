from __future__ import annotations
import regex as re
import json
from typing import Optional

JSON_BLOCK = re.compile(r'\{(?:[^{}]|(?R))*\}', re.DOTALL) # recursive-like via PCRE would be nicer; this is a workable approx

def extract_json(text: str) -> Optional[str]:
    # Try best-effort extraction: first balanced-ish JSON object
    m = JSON_BLOCK.search(text)
    if not m:
        return None
    candidate = m.group(0)
    # Optional: strip trailing commas or fix common issues here
    try:
        _ = json.loads(candidate)
        return candidate
    except Exception:
        return None
