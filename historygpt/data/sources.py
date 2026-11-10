"""Readers for the downloaded corpora in data/raw/ (see scripts/download_corpora.py).

Each source exposes `catalog()`, a DataFrame with one row per work:
    work_id, title, author, year, genre, group, source, license
and `texts(work_ids)`, which yields (work_id, text) for the requested works,
reading the large text columns only for those rows.
"""

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .labels import CLMET_GENRES, author_key, foreign_authors, genre_from_title, is_translation, parse_year

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# A book printed this long after its author died is a reprint of older
# language (e.g. Chaucer printed in 1602) and is dropped.
MAX_POSTHUMOUS_YEARS = 30


def _files(name, pattern="**/*.parquet"):
    return sorted((RAW / name).glob(pattern))


def _death_year(author):
    """Latest year in the first author's catalogue heading, if any ('d. 1364', '1628-1688')."""
    first = re.split(r";", str(author or ""))[0]
    years = [int(y) for y in re.findall(r"1[0-9]\d\d", first)]
    return max(years) if years else None


def _texts_by_id(files, id_col, work_ids, text_col="text"):
    wanted = set(work_ids)
    for f in files:
        table = pq.read_table(f, columns=[id_col, text_col], filters=[(id_col, "in", list(wanted))])
        for wid, text in zip(table[id_col].to_pylist(), table[text_col].to_pylist()):
            yield wid, text


class TCP:
    """EEBO-TCP (1473-1700) and ECCO-TCP (1701-1800): CC0, original spelling."""

    name = "tcp"
    license = "CC0-1.0"
    columns = ["tcp_id", "title", "author", "year", "language", "ecco_proofed"]

    def files(self):
        return _files("eebo") + _files("ecco")

    def catalog(self):
        df = pd.concat(pd.read_parquet(f, columns=self.columns) for f in self.files())
        df = df[df.language == "eng"]
        df = df[df.ecco_proofed.astype(str) != "False"]  # ECCO texts TCP never proofed
        df["year"] = df.year.map(parse_year)
        df = df.dropna(subset=["year"]).astype({"year": int})
        death = df.author.map(_death_year)
        reprint = death.notna() & (df.year - death > MAX_POSTHUMOUS_YEARS)
        df = df[~reprint]
        authors = df.author.map(author_key)
        translated = df.title.map(is_translation) | authors.isin(foreign_authors(df.title, df.author))
        df, authors = df[~translated], authors[~translated]
        return pd.DataFrame({
            "work_id": df.tcp_id,
            "title": df.title,
            "author": df.author,
            "year": df.year,
            "genre": df.title.map(genre_from_title),
            "group": authors.fillna("work:" + df.tcp_id),
            "source": self.name,
            "license": self.license,
        })

    def texts(self, work_ids):
        return _texts_by_id(self.files(), "tcp_id", work_ids)


class CLMET:
    """Corpus of Late Modern English Texts 3.1 (1710-1920): CC BY-NC-SA 4.0."""

    name = "clmet"
    license = "CC-BY-NC-SA-4.0"

    def catalog(self):
        df = pd.read_parquet(_files("clmet")[0], columns=["clmet_id", "title", "author", "year", "genre"])
        df = df[~df.title.map(is_translation)]
        return pd.DataFrame({
            "work_id": df.clmet_id,
            "title": df.title,
            "author": df.author,
            "year": df.year.map(parse_year),
            "genre": df.genre.map(CLMET_GENRES).fillna("prose"),
            "group": df.author.map(author_key).fillna("work:" + df.clmet_id),
            "source": self.name,
            "license": self.license,
        }).dropna(subset=["year"]).astype({"year": int})

    def texts(self, work_ids):
        return _texts_by_id(_files("clmet"), "clmet_id", work_ids)


class Hansard:
    """UK parliamentary debates (1803-1913): Open Parliament Licence v3.0."""

    name = "hansard"
    license = "OGL-UK-3.0 (Open Parliament Licence)"

    def catalog(self):
        frames = [
            pd.read_parquet(f, columns=["doc_id", "title", "year", "is_prose", "duplicate_of", "part", "n_words"])
            for f in _files("hansard")
        ]
        df = pd.concat(frames)
        df = df[df.is_prose & df.duplicate_of.isna() & (df.part == "debates") & (df.n_words >= 150)]
        return pd.DataFrame({
            "work_id": df.doc_id,
            "title": df.title,
            "author": None,
            "year": df.year.astype(int),
            "genre": "parliamentary debate",
            "group": "work:" + df.doc_id,
            "source": self.name,
            "license": self.license,
        })

    def texts(self, work_ids):
        return _texts_by_id(_files("hansard"), "doc_id", work_ids)


class OldBailey:
    """Old Bailey Proceedings, trials (1674-1913): CC BY-NC 4.0."""

    name = "oldbailey"
    license = "CC-BY-NC-4.0"

    def catalog(self):
        frames = [pd.read_parquet(f, columns=["trial_id", "session_id", "date"]) for f in _files("oldbailey")]
        df = pd.concat(frames)
        return pd.DataFrame({
            "work_id": df.trial_id,
            "title": "Old Bailey trial " + df.trial_id,
            "author": None,
            "year": df.date.str[:4].astype(int),
            "genre": "trial record",
            # One court session is printed as one pamphlet: keep it in one split.
            "group": "session:" + df.session_id,
            "source": self.name,
            "license": self.license,
        })

    def texts(self, work_ids):
        for wid, text in _texts_by_id(_files("oldbailey"), "trial_id", work_ids):
            yield wid, _untag_spacing(text)


def _untag_spacing(text):
    """Undo spacing left by the Old Bailey XML tags: 'DUNN , widow ,' and ' ~ ' markers."""
    text = re.sub(r"\s*~\s*", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?)])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text)


SOURCES = {s.name: s for s in (TCP(), CLMET(), Hansard(), OldBailey())}
