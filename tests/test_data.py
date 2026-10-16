import random

import pytest

from historygpt.data.labels import (
    BUCKETS,
    author_key,
    bucket,
    foreign_authors,
    genre_from_title,
    is_translation,
    parse_year,
    split_for,
)
from historygpt.data.passages import MAX_WORDS, MIN_WORDS, candidate_passages, clean, is_clean, n_words, sample_passages

# ~25 words of plain early modern English; repeated to make passages.
SENTENCE = (
    "And when the king was come unto the city, he spake unto the people that were gathered there, "
    "and he told them all that had been done in his absence."
)


def paragraph(sentences):
    return " ".join([SENTENCE] * sentences)


# --- labels -----------------------------------------------------------------


@pytest.mark.parametrize(
    "year, expected",
    [(1500, "1500-1549"), (1549, "1500-1549"), (1666, "1650-1699"), (1899, "1850-1899"), (1499, None), (1900, None), (None, None)],
)
def test_bucket(year, expected):
    assert bucket(year) == expected


def test_buckets_cover_the_period():
    assert BUCKETS[0] == "1500-1549" and BUCKETS[-1] == "1850-1899" and len(BUCKETS) == 8


@pytest.mark.parametrize("value, expected", [("1706 [1705]", 1706), ("[1544?]", 1544), ("ca. 1600", 1600), ("n.d.", None), (None, None)])
def test_parse_year(value, expected):
    assert parse_year(value) == expected


@pytest.mark.parametrize(
    "title, genre",
    [
        ("A sermon preached at Paules Crosse", "sermon"),
        ("The tragedie of Othello, the Moore of Venice", "drama"),
        ("Poems, by J.D.", "poetry"),
        ("The historie of Great Britaine", "history"),
        ("A discourse of the common weal", "treatise"),
        ("Robinson Crusoe", "prose"),
    ],
)
def test_genre_from_title(title, genre):
    assert genre_from_title(title) == genre


@pytest.mark.parametrize(
    "author, key",
    [("Bunyan, John, 1628-1688.", "bunyan, john"), ("Anonymous", None), (None, None), ("Calvin, Jean, 1509-1564.; Golding, Arthur", "calvin, jean")],
)
def test_author_key(author, key):
    assert author_key(author) == key


def test_split_for_is_deterministic_and_roughly_80_10_10():
    splits = [split_for(f"author {i}") for i in range(5000)]
    assert splits == [split_for(f"author {i}") for i in range(5000)]
    shares = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert abs(shares["train"] - 0.8) < 0.03 and abs(shares["val"] - 0.1) < 0.02 and abs(shares["test"] - 0.1) < 0.02


@pytest.mark.parametrize(
    "title",
    [
        "The institution of Christian religion, translated into English by Thomas Norton",
        "Certaine sermons englished by A. Golding",
        "A treatise done into English out of the French",
        "The workes of Seneca, out of the Latine",
        "Letters from the Italian",
    ],
)
def test_is_translation(title):
    assert is_translation(title)


@pytest.mark.parametrize("title", ["The pilgrim's progress", "A discourse of English poetrie", "The Frenche historie", None])
def test_is_not_translation(title):
    assert not is_translation(title)


def test_foreign_authors_learns_mostly_translated_authors():
    titles = ["Sermons, translated out of Latin", "A treatise, englished", "An epistle"] + ["Poems", "A play, translated", "Essays"]
    authors = ["Bullinger, Heinrich, 1504-1575."] * 3 + ["Gascoigne, George"] * 3
    found = foreign_authors(titles, authors)
    assert "bullinger, heinrich" in found  # 2 of 3 marked
    assert "gascoigne, george" not in found  # 1 of 3 marked
    assert "erasmus, desiderius" in found  # hand-listed


# --- passages ---------------------------------------------------------------


def test_clean_strips_markdown_and_apparatus():
    text = "---\ntitle: x\n---\n\n# Heading\n\nSome *emphasis* here.[^1]\n\n[^1]: a note\n\n---\n\n- a bullet [Illustration: a cat]"
    assert clean(text) == ["Some emphasis here.", "a bullet"]


def test_is_clean_accepts_prose():
    assert is_clean(paragraph(6))


@pytest.mark.parametrize(
    "text",
    [
        paragraph(6) + " the 〈◊〉 gap",  # illegible characters in the source
        paragraph(6) + " see https://www.gutenberg.org",
        " ".join(["Gallia est omnis divisa in partes tres quarum unam incolunt Belgae"] * 12),  # Latin
        " ".join(["12 14 16 | 18 20 | 22"] * 40),  # a table
    ],
)
def test_is_clean_rejects_noise(text):
    assert not is_clean(text)


def test_candidate_passages_are_within_length_bounds_and_skip_front_matter():
    front = "FRONT MATTER " + paragraph(6)
    body = [paragraph(i % 5 + 2) for i in range(30)]
    text = "\n\n".join([front] + body + [paragraph(20)])  # one paragraph over MAX_WORDS
    passages = candidate_passages(text)
    assert passages
    assert all(MIN_WORDS <= n_words(p) <= MAX_WORDS for p in passages)
    assert not any("FRONT MATTER" in p for p in passages)


def test_long_paragraph_is_split_on_sentences():
    filler = "\n\n".join(paragraph(2) for _ in range(2))
    text = "\n\n".join([filler, paragraph(30), filler])
    passages = candidate_passages(text)
    assert passages and all(p.rstrip().endswith(".") for p in passages)


def test_sample_passages_keeps_text_order():
    text = "\n\n".join(f"Part {i}. " + paragraph(6) for i in range(40))
    picked = sample_passages(text, 5, random.Random(0))
    order = [int(p.split(".")[0].split()[-1]) for p in picked]
    assert len(picked) == 5 and order == sorted(order)


# --- source readers ----------------------------------------------------------


def test_untag_old_bailey_spacing():
    sources = pytest.importorskip("historygpt.data.sources")
    assert sources._untag_spacing("JOHN DUNN , widow , ~ was indicted .") == "JOHN DUNN, widow, was indicted."


@pytest.mark.parametrize("author, year", [("Bunyan, John, 1628-1688.", 1688), ("Chaucer, Geoffrey, d. 1400", 1400), ("Anonymous", None)])
def test_death_year(author, year):
    sources = pytest.importorskip("historygpt.data.sources")
    assert sources._death_year(author) == year
