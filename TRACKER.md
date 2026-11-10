# Change tracker

A running log of the HistoryGPT rebuild: what is being worked on now, what is
done, and the decisions behind it. Newest entries first within each section.
The plan itself is the [Roadmap](README.md#roadmap) in the README.

## In progress

**Phase 2 — instruction generation** (branch `phase2-dataset`)

Each passage in `data/v2/passages.jsonl` needs a user prompt (e.g. "Write a
sermon on patience, as if written in the 1580s.") so it can be used as a
chat fine-tuning example. Prompts must name the era and genre and describe the
passage's actual topic, or the model learns to ignore the prompt.

- [x] Choose how topics are produced → **LLM labels** (decided 2026-10-16):
  Claude Haiku 5.5 (`claude-haiku-5-5`, low effort, JSON-schema output) reads
  each passage with its title and year and returns a `topic` (modern-English
  phrase) and a `form` ("a sermon", "a witness's testimony at a criminal
  trial"). The form is finer than the title-based genre, which leaves 2,372
  passages as plain "prose". Run through the Message Batches API at half
  price; estimated well under $1 for all 6,000. Passages with no usable label
  fall back to a genre-only prompt.
- [x] `historygpt/data/prompts.py`: 6 topic templates + 3 genre-only
  templates; era named as a decade ("the 1580s"), part of a century ("the late
  16th century") or a year ("around 1583"); choice seeded per passage, so builds
  are reproducible. Chat records use the app's system prompt.
- [x] `scripts/generate_prompts.py` with stages `sample N` (label N passages now
  and print them), `submit` / `collect` (batch), `build` (writes
  `data/v2/chat/{train,val,test}.jsonl`). Labels are cached in
  `data/v2/labels.jsonl`, so nothing is paid for twice.
- [x] Tests for era phrases and chat examples (66 passing); `anthropic==1.15.0`
  added to `requirements-data.txt`. `build` dry-run without labels works.
- [ ] **Blocked: needs `ANTHROPIC_API_KEY` in `.env`**
- [ ] Run `sample 15` and review label quality; adjust instructions if needed
- [ ] `submit` + `collect` for all passages; `build`
- [ ] Spot-check prompts against passages; update README and STATS
- [ ] Commit and push

Notes:
- An unmarked translation found while testing: *Howleglas* (c. 1519, from the
  German Till Eulenspiegel). It's the kind the filter can't see; such cases
  are now flagged by the labelling pass (`translated` field in `labels.jsonl`); whether to drop flagged passages is decided after seeing the counts.
- Setup: the venv is uv-managed and has no pip. Install with
  `uv pip install --python .venv/bin/python -r requirements-data.txt`.

## Done

### 2026-10-16 — Phase 2: dataset v2 without translations

Branch `phase2-dataset`, commit `5826f63`, pushed to GitHub.

- **Corpus readers** (`historygpt/data/sources.py`) for EEBO-TCP and ECCO-TCP
  (CC0), CLMET 3.1 (CC BY-NC-SA), Hansard (Open Parliament Licence) and the Old
  Bailey Proceedings (CC BY-NC). `scripts/download_corpora.py` fetches them
  (~5.5 GB) into `data/raw/`, which is git-ignored.
- **Passage cutting** (`historygpt/data/passages.py`): 120–350-word runs of
  paragraphs from the body of each text (skipping the first 10% and last 5%),
  with Markdown, footnotes and illustrations stripped. Rejects passages with
  illegible-character markers, Gutenberg boilerplate, tables, capitalised title
  pages, or too few English function words (Latin, Welsh).
- **Labels and splits** (`historygpt/data/labels.py`): eight 50-year eras from
  1500 to 1899; coarse genres from titles (sermon, drama, poetry, letter, law,
  news, history, religion, treatise, prose) or CLMET's own genres;
  train/val/test (80/10/10) assigned by a hash of the author, or of the work
  when anonymous, or of the court session for trials, so no author spans two splits.
- **Late reprints dropped**: books printed more than 30 years after the
  author's death (e.g. Chaucer printed in 1602).
- **Translations dropped**. This was v1's main flaw, and it was still in the
  first v2 build: 350 of 5,969 passages came from works whose titles say
  "translated", and 77 from authors like Calvin and Bullinger. A work is now
  dropped if its title has a translation marker, or its author is foreign:
  at least 2 of their works, and at least half, are marked as translated in the
  catalogue (Calvin, Bullinger, Beza, Ovid, Plutarch, Buffon…), or they are on a
  short hand-made list (Erasmus, Luther, Aesop, Montaigne, "France"…). This
  removes 3,466 of 60,572 TCP works (5.7%). A sample of 1,500 surviving title
  pages found about 9 unmarked translations (<1%); scanning title pages was
  rejected because half its hits were false positives.
- **Build** (`scripts/build_dataset.py --per-bucket 750 --seed 670`):
  6,000 passages, 750 per era, 1.38 M words, 4,737 works, 3,057 author groups;
  per-era source mix with trials and debates capped at 15% each. Shortfalls in
  a minor source are made up from the era's main source. Output in
  `data/v2/passages.jsonl` and `data/v2/STATS.md`.
- **Tests**: `tests/test_data.py` (50 tests: eras, years, genres, author keys,
  split balance and determinism, translation detection, passage cleaning and
  bounds). Full suite: 60 passed.
- **README**: dataset section (build commands, sources, licences, split and
  filter rules) and updated repository layout.

### 2026-10-15 — Phase 1: cleanup

Branch `phase1-cleanup`, commit `94af279`, pushed to GitHub.

- OpenAI key read from the environment (`.env`, git-ignored) instead of code;
  `.env.example` added.
- v1 notebooks and data moved to `notebooks/legacy/` and `data/legacy/`; dead
  files removed (`answers_with_ft.ipynb`, `media.json`, duplicate notebooks).
- `backend_gpt.py` replaced by `app.py` (Flask) and `historygpt/llm.py`, the one
  module that calls a model. `POST /api/chat` takes the whole conversation,
  validates it, and adds the system prompt the model was trained with.
- Front end: output rendered as text (no `innerHTML`), conversation history
  kept, send button and Enter key fixed, static paths fixed, unused CDN
  libraries removed.
- Legacy notebook bugs fixed (undefined model variable, per-character printing).
- README with v1's known limitations and the roadmap; pinned requirements;
  pytest suite for the API (10 tests, model stubbed).
