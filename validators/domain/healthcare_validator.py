import json, re

ICD10_MINI = {"A00","A01.0","E11.9","I10","J45.909"}  # extend from refs_icd10_mini.json

def validate_hc_claim_csv(s: str) -> tuple[bool,str]:
    # Very small CSV check (you already have csv_validator; this is domain add-on)
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    if not lines or lines[0] != "patient_id,icd10,procedure,date":
        return False, "bad_header"
    for row in lines[1:]:
        cells = row.split(",")
        if len(cells) != 4: return False, "wrong_row_len"
        pid, icd, proc, date = cells
        if not re.fullmatch(r"PAT-\d{5}", pid): return False, "patient_id_format"
        if icd not in ICD10_MINI: return False, "icd10_invalid"
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date): return False, "date_format"
    return True, "ok"
