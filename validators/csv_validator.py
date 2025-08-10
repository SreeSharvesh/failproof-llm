from __future__ import annotations
import csv, io
from typing import Tuple, Dict, Any

def validate_csv(text: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    if spec.get("strict_no_extra_text", True):
        # fast check: must start with header and contain only CSV-ish lines
        if text.strip().startswith("```"):
            return False, "code_fence_present"
        if "\n\n" in text.strip():
            # often indicates commentary blocks
            pass  # not a hard fail by itself

    try:
        reader = csv.reader(io.StringIO(text.strip()))
        rows = list(reader)
    except Exception as e:
        return False, f"csv_parse_error:{e}"

    if not rows:
        return False, "csv_empty"

    header = rows[0]
    expected_cols = spec.get("columns", [])
    if header != expected_cols:
        return False, f"csv_bad_header:{header}!= {expected_cols}"

    data_rows = rows[1:]
    if len(data_rows) != spec.get("rows", 0):
        return False, f"csv_wrong_row_count:{len(data_rows)}"

    # ensure each row has exact column count
    for r in data_rows:
        if len(r) != len(expected_cols):
            return False, "csv_column_mismatch"

    return True, "ok"
