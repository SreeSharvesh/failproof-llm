from __future__ import annotations
from typing import Tuple, Dict, Any
from html.parser import HTMLParser

class _TagCounter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.counts = {}

    def handle_starttag(self, tag, attrs):
        self.counts[tag] = self.counts.get(tag, 0) + 1

def validate_html(text: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    if spec.get("strict_no_extra_text", True):
        s = text.strip()
        if s.startswith("```") or s.lower().startswith("here is"):
            return False, "html_extraneous_text"

    parser = _TagCounter()
    try:
        parser.feed(text)
    except Exception as e:
        return False, f"html_parse_error:{e}"

    required = spec.get("required_tags", [])
    for tag in required:
        if parser.counts.get(tag, 0) != 1:
            return False, f"html_missing_or_multiple:{tag}={parser.counts.get(tag, 0)}"
    return True, "ok"
