"""Cut clean, self-contained passages out of a full historical text.

Source texts are Markdown (TCP, CLMET) or plain prose (Hansard, Old Bailey).
A passage is a run of consecutive paragraphs, MIN_WORDS..MAX_WORDS long, from
the body of the text, that passes the quality checks in `is_clean`.
"""

import re

MIN_WORDS = 120
MAX_WORDS = 350

# Skip front matter (title pages, dedications) and back matter (indexes, errata).
BODY_START = 0.10
BODY_END = 0.95

# Function words in modern and early modern spelling. Real English prose has
# 35-50% of its tokens in this set; Latin, Welsh or tables score near zero.
FUNCTION_WORDS = frozenset(
    """a an and are as at be bee been but by did do doth for from had hath have haue
    he hee her him his i if in into is it its me my no nor not of on or our shall she
    so than that the thee their them then there they this thou thy to unto vnto upon
    vpon was we wee were what when which who will with would ye you your""".split()
)

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
FOOTNOTE_DEF = re.compile(r"^\[\^\d+\]:.*$", re.M)
FOOTNOTE_REF = re.compile(r"\s*\[\^\d+\]")
ILLUSTRATION = re.compile(r"\[Illustration[^\]]*\]")
EMPHASIS = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
GAP_MARKERS = ("•", "〈", "〉", "")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-z]+")


def clean(text):
    """Strip Markdown and editorial apparatus; return a list of paragraphs."""
    text = FRONT_MATTER.sub("", text).replace("❝", "“").replace("❞", "”")
    text = FOOTNOTE_DEF.sub("", text)
    text = FOOTNOTE_REF.sub("", text)
    text = ILLUSTRATION.sub("", text)
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or set(line) <= set("-*_ "):
                continue  # headings and horizontal rules
            line = re.sub(r"^[-*] ", "", line)  # list bullets
            lines.append(EMPHASIS.sub(r"\1", line))
        if lines:
            paragraphs.append("\n".join(lines))
    return paragraphs


def n_words(text):
    return len(text.split())


def english_ratio(text):
    tokens = WORD.findall(text.lower())
    return sum(t in FUNCTION_WORDS for t in tokens) / len(tokens) if tokens else 0.0


def is_clean(text):
    """True if `text` is readable English prose or verse worth training on."""
    if any(m in text for m in GAP_MARKERS):
        return False  # illegible characters in the source; TCP never guesses them
    if re.search(r"\[pg\s*\d+\]|https?://|gutenberg", text, re.I):
        return False
    chars = [c for c in text if not c.isspace()]
    if sum(c.isalpha() for c in chars) / len(chars) < 0.85:
        return False  # tables, citations, numbered lists
    lines = text.splitlines()
    if sum(not re.search(r"[a-z]", line) for line in lines) > 0.3 * len(lines):
        return False  # title pages and headings in capitals
    return english_ratio(text) >= 0.30


def _split_long(paragraph):
    """Break a paragraph longer than MAX_WORDS into sentence-aligned pieces."""
    pieces, current = [], []
    for sentence in SENTENCE_END.split(paragraph):
        if current and n_words(" ".join(current + [sentence])) > MAX_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if n_words(p) <= MAX_WORDS]


def candidate_passages(text):
    """All non-overlapping clean passages from the body of `text`, in order."""
    paragraphs = clean(text)
    start, end = int(len(paragraphs) * BODY_START), int(len(paragraphs) * BODY_END)
    units = []
    for p in paragraphs[start:max(end, start + 1)]:
        units.extend(_split_long(p) if n_words(p) > MAX_WORDS else [p])

    passages, current = [], []
    for unit in units:
        if current and n_words("\n\n".join(current + [unit])) > MAX_WORDS:
            current = []  # would overflow: drop the short run and start fresh
        current.append(unit)
        passage = "\n\n".join(current)
        if n_words(passage) >= MIN_WORDS:
            if is_clean(passage):
                passages.append(passage)
            current = []
    return passages


def sample_passages(text, k, rng):
    """Up to `k` clean passages from `text`, chosen at random but kept in text order."""
    candidates = candidate_passages(text)
    if len(candidates) <= k:
        return candidates
    return [candidates[i] for i in sorted(rng.sample(range(len(candidates)), k))]
