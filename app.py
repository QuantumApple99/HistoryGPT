import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from historygpt import llm

MAX_MESSAGES = 20
MAX_CHARS = 4000

log = logging.getLogger(__name__)


def validate_messages(messages):
    """Return an error string, or None if `messages` is a valid chat history."""
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list"
    if len(messages) > MAX_MESSAGES:
        return f"at most {MAX_MESSAGES} messages are allowed"
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            return "each message needs exactly 'role' and 'content'"
        if m["role"] not in ("user", "assistant"):
            return "role must be 'user' or 'assistant'"
        if not isinstance(m["content"], str) or not m["content"].strip():
            return "content must be a non-empty string"
        if len(m["content"]) > MAX_CHARS:
            return f"each message is limited to {MAX_CHARS} characters"
    if messages[-1]["role"] != "user":
        return "the last message must come from the user"
    return None


def create_app(generate=llm.generate):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/chat")
    def chat():
        body = request.get_json(silent=True) or {}
        messages = body.get("messages")
        error = validate_messages(messages)
        if error:
            return jsonify({"error": error}), 400
        try:
            reply = generate(messages)
        except Exception:
            log.exception("model call failed")
            return jsonify({"error": "The model could not be reached. Try again shortly."}), 502
        return jsonify({"message": reply})

    return app


if __name__ == "__main__":
    load_dotenv()
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
