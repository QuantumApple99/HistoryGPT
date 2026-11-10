"""Download the source corpora for dataset v2 into data/raw/ (about 5.5 GB).

Usage: python scripts/download_corpora.py [name ...]
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# name -> (Hugging Face dataset repo, files to fetch)
CORPORA = {
    "eebo": ("Vintage-LLM/EEBO", ["data/*.parquet", "README.md"]),
    "ecco": ("Vintage-LLM/ECCO", ["data/*.parquet", "README.md"]),
    "clmet": ("Vintage-LLM/CLMET", ["data/*.parquet", "README.md"]),
    "hansard": ("Vintage-LLM/Hansard", ["data/*/*.parquet", "README.md"]),
    "oldbailey": ("biglam/old_bailey_proceedings", ["trials/*.parquet", "README.md"]),
}


def main(names):
    for name in names or CORPORA:
        repo, patterns = CORPORA[name]
        print(f"downloading {name} ({repo})", flush=True)
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=RAW / name,
        )
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
