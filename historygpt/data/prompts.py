"""Turn a labelled passage into a chat training example.

Each passage carries a `topic` and a `form` (written by an LLM in
scripts/generate_prompts.py). The user prompt asks for that form, on that
topic, from the passage's period; the passage itself is the reply. Wording and
the way the period is named vary, chosen per passage from a seeded RNG, so the
model learns the request, not one template.
"""

import random

from historygpt.llm import SYSTEM_PROMPT

# Used when a passage has no LLM label (e.g. the request was refused).
GENRE_FORMS = {
    "sermon": "a sermon",
    "drama": "a scene from a play",
    "poetry": "a poem",
    "letter": "a letter",
    "law": "a legal text",
    "news": "a news report",
    "history": "a passage of history",
    "religion": "a devotional text",
    "treatise": "a passage from a treatise",
    "fiction": "a passage from a novel",
    "trial record": "an account of a criminal trial",
    "parliamentary debate": "a speech in Parliament",
    "prose": "a passage of prose",
}

TOPIC_TEMPLATES = [
    "Write {form} about {topic}, as if written in {era}.",
    "Write {form} on {topic} in the English of {era}.",
    "In the style of {era}, write {form} about {topic}.",
    "Compose {form} from {era} about {topic}.",
    "Imagine it is {era}. Write {form} about {topic}.",
    "Write about {topic} as {form} from {era}.",
]

FORM_TEMPLATES = [
    "Write {form}, as if written in {era}.",
    "Write {form} in the English of {era}.",
    "Compose {form} from {era}.",
]


def _ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def era_phrases(year):
    """Ways of naming the period of `year`: decade, part of century, approximate year."""
    century = _ordinal(year // 100 + 1)
    part = "early" if year % 100 < 33 else "mid" if year % 100 < 67 else "late"
    return [f"the {year // 10 * 10}s", f"the {part} {century} century", f"around {year}"]


def user_prompt(passage, label, rng):
    """The request whose answer is `passage["text"]`. `label` may be None."""
    era = rng.choice(era_phrases(passage["year"]))
    topic = (label or {}).get("topic")
    form = (label or {}).get("form") or GENRE_FORMS.get(passage["genre"], "a passage of prose")
    template = rng.choice(TOPIC_TEMPLATES if topic else FORM_TEMPLATES)
    text = template.format(form=form, topic=topic, era=era)
    return text[0].upper() + text[1:]


def chat_example(passage, label, seed):
    """One fine-tuning record in OpenAI/HF chat format, plus metadata for analysis."""
    rng = random.Random(f"{seed}:{passage['id']}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(passage, label, rng)},
            {"role": "assistant", "content": passage["text"]},
        ],
        "id": passage["id"],
        "year": passage["year"],
        "bucket": passage["bucket"],
        "genre": passage["genre"],
        "source": passage["source"],
        "labelled": bool(label and label.get("topic")),
    }
