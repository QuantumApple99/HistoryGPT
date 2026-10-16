"""Build dataset v2: era-balanced passages from period sources.

Usage: python scripts/build_dataset.py [--per-bucket 750] [--seed 670] [--out data/v2]

Writes <out>/passages.jsonl (one passage per line, with its split) and
<out>/STATS.md. The build is deterministic for a given seed.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from historygpt.data.labels import BUCKETS, bucket, split_for  # noqa: E402
from historygpt.data.passages import candidate_passages, n_words  # noqa: E402
from historygpt.data.sources import SOURCES  # noqa: E402

# Share of each era's passages drawn from each source. Books carry the main
# literary and religious record; trials and debates add speech-like registers
# where they exist, capped so they cannot dominate an era.
MIX = {
    "1500-1549": {"tcp": 1.0},
    "1550-1599": {"tcp": 1.0},
    "1600-1649": {"tcp": 1.0},
    "1650-1699": {"tcp": 0.85, "oldbailey": 0.15},
    "1700-1749": {"tcp": 0.75, "clmet": 0.10, "oldbailey": 0.15},
    "1750-1799": {"tcp": 0.70, "clmet": 0.15, "oldbailey": 0.15},
    "1800-1849": {"clmet": 0.70, "hansard": 0.15, "oldbailey": 0.15},
    "1850-1899": {"clmet": 0.70, "hansard": 0.15, "oldbailey": 0.15},
}
assert set(MIX) == set(BUCKETS) and all(abs(sum(m.values()) - 1) < 1e-9 for m in MIX.values())

# Secondary sources are sampled first; any passages they cannot supply are
# added to the quota of the era's main (largest-share) source, read later.
SOURCE_ORDER = ["oldbailey", "hansard", "clmet", "tcp"]
assert all(max(m, key=m.get) in SOURCE_ORDER[SOURCE_ORDER.index(s) :] for m in MIX.values() for s in m)

POOL_FACTOR = 2  # works read per passage needed; spare works cover ones with no clean passage
MAX_PER_WORK = 10


def quotas(per_bucket):
    out = {}
    for b, mix in MIX.items():
        shares = {s: int(per_bucket * share) for s, share in mix.items()}
        shares[max(mix, key=mix.get)] += per_bucket - sum(shares.values())
        out[b] = shares
    return out


def fill(pool, candidates, quota, rng, max_per_group):
    """Pick `quota` passages, spreading them over as many works and groups as possible."""
    for wid in pool.work_id:
        rng.shuffle(candidates.get(wid, []))
    taken, per_group, picked = Counter(), Counter(), []
    for cap in range(1, MAX_PER_WORK + 1):
        for row in pool.itertuples():
            if len(picked) == quota:
                return picked
            cands = candidates.get(row.work_id, [])
            if taken[row.work_id] < min(cap, len(cands)) and per_group[row.group] < max_per_group:
                picked.append((row, cands[taken[row.work_id]]))
                taken[row.work_id] += 1
                per_group[row.group] += 1
    return picked


def build(per_bucket, seed):
    rng = random.Random(seed)
    targets = quotas(per_bucket)
    max_per_group = max(5, per_bucket // 40)
    records = []

    for name in SOURCE_ORDER:
        source = SOURCES[name]
        catalog = source.catalog()
        catalog["bucket"] = catalog.year.map(bucket)
        pools = {}
        for b, shares in targets.items():
            if name not in shares:
                continue
            works = catalog[catalog.bucket == b].sample(frac=1, random_state=rng.randrange(2**32))
            pools[b] = works.head(max(shares[name] * POOL_FACTOR, 200))
        wanted = pd.concat(pools.values()).work_id.tolist() if pools else []
        print(f"{name}: reading {len(wanted)} works", flush=True)

        candidates = {wid: candidate_passages(text) for wid, text in source.texts(wanted)}

        for b, pool in pools.items():
            quota = targets[b][name]
            picked = fill(pool, candidates, quota, rng, max_per_group)
            if len(picked) < quota:
                main = max(MIX[b], key=MIX[b].get)
                print(f"  {b}: {name} gave {len(picked)}/{quota}; the rest go to {main}", flush=True)
                if main != name:
                    targets[b][main] += quota - len(picked)
            for row, text in picked:
                records.append({
                    "text": text,
                    "year": int(row.year),
                    "bucket": b,
                    "genre": row.genre,
                    "source": name,
                    "work_id": row.work_id,
                    "title": row.title,
                    "author": row.author,
                    "group": row.group,
                    "split": split_for(row.group),
                    "license": row.license,
                    "n_words": n_words(text),
                })

    records.sort(key=lambda r: (BUCKETS.index(r["bucket"]), r["source"], r["work_id"]))
    for i, r in enumerate(records):
        r["id"] = f"hgpt2-{i:05d}"
    return records


def stats_markdown(df, per_bucket, seed):
    def table(frame):
        return frame.to_markdown() if hasattr(frame, "to_markdown") else f"```\n{frame}\n```"

    lines = [
        "# Dataset v2 statistics",
        "",
        f"Built with `--per-bucket {per_bucket} --seed {seed}`: "
        f"{len(df)} passages, {df.n_words.sum():,} words, "
        f"{df.work_id.nunique()} works, {df.group.nunique()} author/work groups.",
        "",
        "## Passages by era and source",
        "",
        table(pd.crosstab(df.bucket, df.source, margins=True)),
        "",
        "## Passages by era and split",
        "",
        table(pd.crosstab(df.bucket, df.split, margins=True)),
        "",
        "## Passages by genre",
        "",
        table(df.genre.value_counts().to_frame("passages")),
        "",
        "## Passage length (words)",
        "",
        table(df.groupby("bucket").n_words.describe()[["min", "50%", "max"]].round(0)),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-bucket", type=int, default=750)
    parser.add_argument("--seed", type=int, default=670)
    parser.add_argument("--out", type=Path, default=Path("data/v2"))
    args = parser.parse_args()

    records = build(args.per_bucket, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "passages.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    leaks = df.groupby("group").split.nunique()
    assert (leaks == 1).all(), "an author/work group spans more than one split"
    (args.out / "STATS.md").write_text(stats_markdown(df, args.per_bucket, args.seed), encoding="utf-8")
    print(f"wrote {len(records)} passages to {args.out}")
    print(pd.crosstab(df.bucket, df.split, margins=True))


if __name__ == "__main__":
    main()
