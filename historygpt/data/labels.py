"""Era buckets, coarse genre labels, and leakage-safe split assignment."""

import hashlib
import re

FIRST_YEAR, LAST_YEAR, BUCKET_YEARS = 1500, 1899, 50


def bucket(year):
    """'1650-1699' for 1666; None outside FIRST_YEAR..LAST_YEAR."""
    if year is None or not FIRST_YEAR <= year <= LAST_YEAR:
        return None
    start = year - (year - FIRST_YEAR) % BUCKET_YEARS
    return f"{start}-{start + BUCKET_YEARS - 1}"


BUCKETS = [bucket(y) for y in range(FIRST_YEAR, LAST_YEAR + 1, BUCKET_YEARS)]


def parse_year(value):
    """First plausible four-digit year in a catalogue date such as '1706 [1705]'."""
    m = re.search(r"1[4-9]\d\d", str(value or ""))
    return int(m.group()) if m else None


# Checked in order against the lower-cased title; the first match wins.
TITLE_GENRES = [
    ("sermon", r"\bsermons?\b|preached|homil"),
    ("drama", r"\b(comedy|comedie|tragedy|tragedie|tragi-?comedy|interlude|masque|farce|opera|oratorio|burletta|drama|play)\b|as it (is|was) (acted|performed)|in (one|two|three|four|five) acts"),
    ("poetry", r"\b(poems?|poeme|verses?|ballads?|songs?|ditt(y|ie)|sonnets?|elegy|elegie|ode|epigrams?|satyr|pastoral)\b|\bin (english )?(heroic )?verse\b"),
    ("letter", r"\b(letters?|epistles?)\b"),
    ("law", r"^(an )?act\b|\bacts? (of|for|to)\b|\b(statutes?|proclamation|ordinance|by the king|by the queen)\b"),
    ("news", r"\b(news|newes|intelligence|true report)\b|\b(a|true|full|brief|briefe) relation\b|account of the late"),
    ("history", r"\b(history|historie|chronicles?|annals|life and death|lives of)\b"),
    ("religion", r"\b(catechism|prayers?|divinity|scripture|gospel|psalms?|meditations?)\b"),
    ("treatise", r"\b(treatise|discourse|essay|inquiry|enquiry|considerations|observations|dialogue)\b"),
]

CLMET_GENRES = {
    "Narrative fiction": "fiction",
    "Narrative non-fiction": "history",
    "Drama": "drama",
    "Letters (LET)": "letter",
    "Treatise": "treatise",
}


def genre_from_title(title):
    t = (title or "").lower()
    for genre, pattern in TITLE_GENRES:
        if re.search(pattern, t):
            return genre
    return "prose"


def author_key(author):
    """Normalise a catalogue author ('Bunyan, John, 1628-1688.') to 'bunyan, john'."""
    if not author or str(author).strip().lower() in ("nan", "none", "anonymous", "anon"):
        return None
    first = re.split(r"[;.]", str(author))[0]
    name = [p.strip() for p in first.split(",") if p.strip() and not re.search(r"\d", p)]
    return ", ".join(name[:2]).lower() or None


# Translations teach the target language's style at the translator's date, but
# catalogues file them under the original author. Titles usually say so.
_LANGS = r"french|frenche|latin|latine|latyn|italian|italien|spanish|spanishe|dutch|duch|german|greek|greke|hebrew|portugu"
TRANSLATION = re.compile(
    r"translat|englished|\b(done|turned|put|rendred|rendered|made) (in)?to englishe?\b|\bin englishe\b"
    rf"|\bout of (the )?({_LANGS})|\bfrom the ({_LANGS}|original)\b",
    re.I,
)

# Authors (and corporate headings) whose English editions are translations but
# fall below the catalogue-wide rate in `foreign_authors`.
KNOWN_FOREIGN = frozenset({
    "france", "spain", "netherlands", "united provinces of the netherlands", "holy roman empire",
    "catholic church", "louis xiv, king of france", "eglises réformées de france",
    "erasmus, desiderius", "luther, martin", "aesop", "aristotle", "plato", "josephus, flavius",
    "montaigne, michel de", "cervantes saavedra, miguel de", "guevara, antonio de", "tasso, torquato",
    "zwingli, ulrich", "musculus, wolfgang", "ochino, bernardino", "hemmingsen, niels",
    "ursinus, zacharias", "pascal, blaise", "fénelon, françois de salignac de la mothe-",
    "voltaire", "rousseau, jean-jacques", "molière",
})

# An author is treated as foreign when at least this many of their works, and
# this share of them, have a translation marker in the title.
FOREIGN_MIN_MARKED, FOREIGN_MIN_SHARE = 2, 0.5


def is_translation(title):
    return bool(TRANSLATION.search(title or ""))


def foreign_authors(titles, authors):
    """Author keys whose catalogued works are mostly marked as translations."""
    marked, total = {}, {}
    for title, author in zip(titles, authors):
        key = author_key(author)
        if key:
            total[key] = total.get(key, 0) + 1
            marked[key] = marked.get(key, 0) + is_translation(title)
    learned = {k for k, n in total.items() if marked[k] >= FOREIGN_MIN_MARKED and marked[k] / n >= FOREIGN_MIN_SHARE}
    return learned | KNOWN_FOREIGN


SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def split_for(group):
    """Deterministic split for a group key (an author, or a work if anonymous).

    Grouping by author keeps every work by one writer, and every edition of
    one work, in a single split, so the test set measures style, not recall.
    """
    h = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, share in SPLITS:
        cumulative += share
        if h < cumulative:
            return name
    return SPLITS[-1][0]
