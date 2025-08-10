import json, re

FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE
)
BRACE_CANDIDATE_RE = re.compile(r"[\{\[]")

def _balanced_json_slice(text: str) -> str | None:
    """
    Scan for the largest balanced JSON object/array and return the slice.
    Works even if there is leading/trailing prose.
    """
    s = text
    n = len(s)
    best = None
    i = 0
    while True:
        m = BRACE_CANDIDATE_RE.search(s, i)
        if not m:
            break
        start = m.start()
        open_ch = s[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        j = start
        in_string = False
        esc = False
        while j < n:
            ch = s[j]
            if in_string:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        # candidate slice
                        candidate = s[start : j + 1]
                        try:
                            json.loads(candidate)
                            # prefer the largest valid slice
                            if not best or len(candidate) > len(best):
                                best = candidate
                        except Exception:
                            pass
                        break
            j += 1
        i = start + 1
    return best

def extract_json(text: str) -> str | None:
    """
    Robustly extract a JSON object/array from mixed prose+JSON outputs.
    Tries, in order:
      1) fenced code blocks ```json ... ``` / ``` ... ```
      2) the largest balanced JSON object/array in the whole text
      3) a light cleanup pass to fix common fence/label noise
    Returns a JSON string or None.
    """
    if not text or not text.strip():
        return None

    # 1) Prefer fenced code blocks
    for m in FENCE_RE.finditer(text):
        block = m.group(1).strip()
        # Some models echo 'json' or headings inside the fence; strip leading labels
        block = re.sub(r"^\s*(json|JSON)\s*$", "", block)
        try:
            json.loads(block)
            return block
        except Exception:
            # try balanced scan within the block as fallback
            inner = _balanced_json_slice(block)
            if inner:
                return inner

    # 2) Try balanced scan in the whole text
    candidate = _balanced_json_slice(text)
    if candidate:
        return candidate

    # 3) Light cleanup: remove markdown headings and retry
    cleaned = re.sub(r"^#+\s.*?$", "", text, flags=re.MULTILINE)
    cleaned = cleaned.replace("```json", "```")
    m = FENCE_RE.search(cleaned)
    if m:
        block = m.group(1).strip()
        try:
            json.loads(block)
            return block
        except Exception:
            inner = _balanced_json_slice(block)
            if inner:
                return inner

    # Nothing worked
    return None
