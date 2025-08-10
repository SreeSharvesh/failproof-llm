def validate_legal_html_snippet(s: str) -> tuple[bool,str]:
    # After your html_validator says structure is ok, ensure simple clause cues
    low = s.lower()
    # very light heuristic: must mention at least one legal cue word
    cues = ["confidential", "governing law", "non-compete", "noncompete", "assignment", "term and termination"]
    if not any(c in low for c in cues):
        return False, "no_legal_cue"
    return True, "ok"
