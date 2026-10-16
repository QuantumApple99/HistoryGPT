# HistoryGPT

A language model that writes in the style of English from a chosen period
(roughly 1500–1900), plus a small web app to chat with it.

> **Status: being rebuilt.** The first version fine-tuned GPT-3.5 on 82
> hand-picked passages. The next version moves to a LoRA fine-tune of an
> open-weight model, a larger period-balanced corpus, and a real evaluation.
> See [Roadmap](#roadmap).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add your OPENAI_API_KEY
python app.py                 # http://127.0.0.1:5000
```

Run the tests with `pytest`.

## Repository layout

```
app.py                 Flask app: GET / (chat page), POST /api/chat
historygpt/llm.py      The only module that calls a model (swap point for the LoRA model)
templates/, static/    Chat front end
tests/                 API tests (no network; the model is stubbed)
data/legacy/           v1 dataset: raw passages, train/validation JSONL, training metrics
notebooks/legacy/      v1 notebooks: data conversion, format checks, fine-tuning, comparison
```

### API

`POST /api/chat` with the full conversation so far:

```json
{"messages": [{"role": "user", "content": "Write a letter about the plague, as if it was written in the 17th century."}]}
```

Returns `{"message": "..."}`, or `{"error": "..."}` with status 400 (bad input)
or 502 (model unavailable). Roles are `user` and `assistant`; the server adds
the system prompt itself.

## v1 (legacy) and its limitations

v1 fine-tuned `gpt-3.5-turbo-0125` for 5 epochs on 82 prompt→passage pairs
(19 for validation), drawn from 101 excerpts in `data/legacy/Dataset_RAW.txt`.
Known problems, which v2 is designed to fix:

- **Translations labelled with the original's date.** The Prince, Utopia, the
  Decameron, Don Quixote and others are modern English translations, so the
  model learned 19th/20th-century prose tagged as earlier centuries.
- **Imbalanced eras.** 70 of 82 training examples are 16th–17th century; the
  18th–19th centuries have 6 between them, and a few sources are older than the
  "500 years" scope.
- **Train/validation leakage.** Several works appear in both splits (e.g.
  consecutive lines of Othello).
- **No evaluation** beyond the fine-tuning loss, which is too noisy at this
  size to show whether period style was learned.

## Roadmap

1. **Cleanup** ✅ secrets out of code, repo layout, web app fixes, tests.
2. **Dataset v2** — period-balanced corpus of texts originally written in
   English (e.g. EEBO-TCP, ECCO-TCP, Project Gutenberg, Old Bailey
   Proceedings) with verified dates; generated instructions; splits by work
   and author.
3. **Evaluation** — era classifier, anachronism rate, prompt adherence, blind
   human comparison; prompting and retrieval baselines.
4. **Model** — LoRA fine-tune of an open-weight model, compared against the
   baselines.
5. **App v2** — era and genre controls, modern→period style transfer,
   anachronism highlighting, similar real passages, streaming, deployment.
