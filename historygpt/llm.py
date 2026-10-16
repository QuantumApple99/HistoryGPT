"""Model access for HistoryGPT.

Everything that talks to a model goes through `generate`, so swapping the
OpenAI fine-tune for a local LoRA model only touches this module.
"""

import os

# Must match the system prompt used in data/legacy/*.jsonl, or the fine-tune
# is queried out of distribution.
SYSTEM_PROMPT = (
    "HistoryGPT is a chatbot that generates linguistically accurate "
    "historical text from the past 500 years."
)

DEFAULT_MODEL = "ft:gpt-3.5-turbo-0125:personal:historygpt:9HnX5Uv9"


def generate(messages):
    """Return the assistant reply for a list of {"role", "content"} messages."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.chat.completions.create(
        model=os.environ.get("HISTORYGPT_MODEL", DEFAULT_MODEL),
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    )
    return response.choices[0].message.content
