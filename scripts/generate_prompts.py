"""Label each dataset v2 passage with a topic and form, then write chat examples.

Usage:
    python scripts/generate_prompts.py sample [N]   # label N random passages now and print them
    python scripts/generate_prompts.py submit       # send every unlabelled passage as one batch
    python scripts/generate_prompts.py collect      # wait for the batch and store its labels
    python scripts/generate_prompts.py build        # write data/v2/chat/{train,val,test}.jsonl

Labels come from Claude Haiku via the Message Batches API (half price) and are
cached in data/v2/labels.jsonl, so each passage is labelled once. Needs
ANTHROPIC_API_KEY in .env for every stage except `build`.
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from historygpt.data.prompts import chat_example  # noqa: E402

DATA = ROOT / "data" / "v2"
PASSAGES = DATA / "passages.jsonl"
LABELS = DATA / "labels.jsonl"
BATCH_STATE = DATA / "labels_batch.json"
CHAT = DATA / "chat"
SEED = 670

MODEL = "claude-haiku-5-5"

INSTRUCTIONS = """You label passages of historical English (1500-1899) for a dataset that teaches a model to write in period style on request. The passage may use early modern spelling.

Return:
- topic: what this passage is about, as a short phrase in modern English (3-12 words), starting lower case unless it begins with a proper noun, with no final full stop. Describe the subject matter so that someone could ask for a passage like this one: "a sermon about {topic}". Be specific to this passage rather than the whole work. Do not mention the period, the style, or that the text is old.
- form: what kind of writing this is, as a short noun phrase with an article, specific to this passage. Examples: "a sermon", "a letter to a friend", "a dialogue from a comedy", "a witness's testimony at a criminal trial", "a speech in Parliament", "a passage from a novel", "a royal proclamation", "a moral essay".
- translated: true if this is probably an English translation of a work first written in another language (the title or text says so, or it is a known foreign work such as a classical author, a continental reformer, or a European folk tale); otherwise false."""

SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}, "form": {"type": "string"}, "translated": {"type": "boolean"}},
    "required": ["topic", "form", "translated"],
    "additionalProperties": False,
}


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def labels_by_id():
    return {r["id"]: r for r in read_jsonl(LABELS)}


def request_params(passage):
    return {
        "model": MODEL,
        # Haiku thinks by default and thinking counts toward max_tokens.
        "max_tokens": 2048,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "system": INSTRUCTIONS,
        "messages": [{
            "role": "user",
            "content": f"Title of the work: {passage['title']}\nYear: {passage['year']}\n\n<passage>\n{passage['text']}\n</passage>",
        }],
    }


def parse_label(passage_id, message):
    """A label record from a response, or one with an error if there is no usable answer."""
    record = {"id": passage_id, "model": MODEL}
    if message.stop_reason != "end_turn":
        return {**record, "error": f"stop_reason={message.stop_reason}"}
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {**record, "error": "invalid json"}
    topic, form = data["topic"].strip().rstrip("."), data["form"].strip().rstrip(".")
    if not topic or not form:
        return {**record, "error": "empty label"}
    form = form[0].lower() + form[1:]  # used mid-sentence
    return {**record, "topic": topic, "form": form, "translated": bool(data["translated"])}


def client():
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return anthropic.Anthropic()


def sample(n):
    passages = random.Random(SEED).sample(read_jsonl(PASSAGES), n)
    c, labels = client(), []
    for p in passages:
        label = parse_label(p["id"], c.messages.create(**request_params(p)))
        labels.append(label)
        example = chat_example(p, label, SEED)
        print(f"--- {p['id']} {p['year']} {p['source']}/{p['genre']} | {p['title'][:70]}")
        flag = " [translated?]" if label.get("translated") else ""
        print(f"    label:  {label.get('form')} | {label.get('topic')}{flag} {label.get('error', '')}")
        print(f"    prompt: {example['messages'][1]['content']}")
        print(f"    text:   {p['text'][:160]!r}")
    known = labels_by_id()
    append_jsonl(LABELS, [lab for lab in labels if "error" not in lab and lab["id"] not in known])


def submit():
    from anthropic.types.messages.batch_create_params import Request

    if BATCH_STATE.exists():
        sys.exit(f"a batch is already pending ({BATCH_STATE}); run `collect` first")
    known = labels_by_id()
    todo = [p for p in read_jsonl(PASSAGES) if p["id"] not in known]
    if not todo:
        print("every passage is already labelled")
        return
    batch = client().messages.batches.create(
        requests=[Request(custom_id=p["id"], params=request_params(p)) for p in todo]
    )
    BATCH_STATE.write_text(json.dumps({"batch_id": batch.id, "requests": len(todo)}))
    print(f"submitted batch {batch.id} with {len(todo)} passages")


def collect():
    state = json.loads(BATCH_STATE.read_text())
    c = client()
    while (batch := c.messages.batches.retrieve(state["batch_id"])).processing_status != "ended":
        counts = batch.request_counts
        print(f"{batch.processing_status}: {counts.succeeded} done, {counts.processing} processing", flush=True)
        time.sleep(60)

    known, new, failed = labels_by_id(), [], []
    for result in c.messages.batches.results(state["batch_id"]):
        if result.custom_id in known:
            continue
        if result.result.type == "succeeded":
            label = parse_label(result.custom_id, result.result.message)
        else:
            label = {"id": result.custom_id, "model": MODEL, "error": result.result.type}
        (failed if "error" in label else new).append(label)
    append_jsonl(LABELS, new)
    BATCH_STATE.unlink()
    print(f"stored {len(new)} labels; {len(failed)} failed (these fall back to genre-only prompts)")
    for f in failed[:20]:
        print(f"  {f['id']}: {f['error']}")


def build():
    labels = labels_by_id()
    by_split = {}
    for p in read_jsonl(PASSAGES):
        by_split.setdefault(p["split"], []).append(chat_example(p, labels.get(p["id"]), SEED))
    CHAT.mkdir(exist_ok=True)
    for split, examples in sorted(by_split.items()):
        with open(CHAT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        n_labelled = sum(e["labelled"] for e in examples)
        print(f"{split}: {len(examples)} examples ({n_labelled} with topics) -> {CHAT / f'{split}.jsonl'}")
    flagged = sum(bool(lab.get("translated")) for lab in labels.values())
    print(f"{flagged} passages flagged as possible translations (kept; see labels.jsonl)")


if __name__ == "__main__":
    stage, *rest = sys.argv[1:] or ["help"]
    if stage == "sample":
        sample(int(rest[0]) if rest else 10)
    elif stage in ("submit", "collect", "build"):
        globals()[stage]()
    else:
        sys.exit(__doc__)
