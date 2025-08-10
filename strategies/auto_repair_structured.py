from __future__ import annotations
import io, re, csv
from html.parser import HTMLParser

# ---------- Common helpers ----------

CODE_FENCE = re.compile(r"^```.*?$|```$", re.MULTILINE)
TRIPLE_BACKTICKS_BLOCK = re.compile(r"```(.*?)```", re.DOTALL)
EXTRANEOUS_PREFIX = re.compile(r"^\s*(here is|output|only|note:|example[:]?)[^\n]*\n", re.IGNORECASE)

def strip_markdown_and_explanations(text: str) -> str:
    s = text.strip()
    # remove code fences entirely
    s = CODE_FENCE.sub("", s)
    # drop common “Here is …” lead-ins
    s = EXTRANEOUS_PREFIX.sub("", s)
    return s.strip()

def normalize_homoglyphs(s: str) -> str:
    # minimal: fix 'currеncy' (cyrillic e) -> 'currency'; add more if needed
    return s.replace("currеncy", "currency")

# ---------- CSV repair ----------

def try_extract_csv_block(text: str, expected_header: list[str]) -> str | None:
    """
    Heuristic: find the line with the exact expected header, then take that
    line + following lines until a blank line or non-CSV-looking line.
    """
    lines = [l.rstrip("\r") for l in text.splitlines() if l.strip()]
    header_line = ",".join(expected_header)
    if header_line in lines:
        idx = lines.index(header_line)
        block = [lines[idx]]
        for j in range(idx+1, len(lines)):
            # break if the row doesn't have expected # of commas
            if lines[j].count(",") != len(expected_header)-1:
                break
            block.append(lines[j])
        return "\n".join(block)
    return None

def repair_csv(text: str, expected_header: list[str], rows: int) -> str | None:
    s = strip_markdown_and_explanations(text)
    s = normalize_homoglyphs(s)

    # direct parse first
    if _csv_parses(s, expected_header, rows):
        return s

    # extract a plausible CSV block around the header
    block = try_extract_csv_block(s, expected_header)
    if block and _csv_parses(block, expected_header, rows):
        return block

    # last resort: grab first fenced block content
    m = TRIPLE_BACKTICKS_BLOCK.search(text)
    if m:
        candidate = strip_markdown_and_explanations(m.group(1))
        candidate = normalize_homoglyphs(candidate)
        if _csv_parses(candidate, expected_header, rows):
            return candidate

    return None

def _csv_parses(text: str, expected_header: list[str], rows: int) -> bool:
    try:
        rdr = csv.reader(io.StringIO(text.strip()))
        all_rows = list(rdr)
    except Exception:
        return False
    if not all_rows:
        return False
    if all_rows[0] != expected_header:
        return False
    data = all_rows[1:]
    if len(data) != rows:
        return False
    for r in data:
        if len(r) != len(expected_header):
            return False
    return True

# ---------- HTML repair ----------

class _TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.title = None
        self.h1 = None
        self.p = None
        self.current = None

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)
        if tag in ("title","h1","p"):
            self.current = tag

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        self.current = None

    def handle_data(self, data):
        if self.current == "title" and self.title is None:
            self.title = (self.title or "") + data.strip()
        elif self.current == "h1" and self.h1 is None:
            self.h1 = (self.h1 or "") + data.strip()
        elif self.current == "p" and self.p is None:
            self.p = (self.p or "") + data.strip()

def repair_html(text: str) -> str | None:
    s = strip_markdown_and_explanations(text)
    parser = _TagCollector()
    try:
        parser.feed(s)
    except Exception:
        # try to strip fences and feed again
        s2 = strip_markdown_and_explanations(TRIPLE_BACKTICKS_BLOCK.sub(r"\1", s))
        parser = _TagCollector()
        try:
            parser.feed(s2)
        except Exception:
            return None

    title = parser.title or "Title"
    h1 = parser.h1 or "Heading"
    p = parser.p or "Paragraph."

    # Rebuild a canonical minimal snippet with exactly one of each
    snippet = f"<title>{title}</title>\n<h1>{h1}</h1>\n<p>{p}</p>"
    return snippet
