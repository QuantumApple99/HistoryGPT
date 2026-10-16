// Chat history sent to the server on every turn, so the model sees the conversation.
const history = [];

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

function addMessage(role, text) {
    const message = document.createElement("div");
    message.className = "message";

    const avatar = document.createElement("div");
    avatar.className = "user";
    const img = document.createElement("img");
    img.src = role === "user" ? "/static/img/user.png" : "/static/img/gpt.png";
    img.alt = role === "user" ? "You" : "HistoryGPT";
    avatar.appendChild(img);

    const content = document.createElement("div");
    content.className = "content";
    const p = document.createElement("p");
    p.style.whiteSpace = "pre-wrap";
    // textContent, never innerHTML: model output must not be able to inject markup.
    p.textContent = text;
    content.appendChild(p);

    message.append(avatar, content);
    messagesEl.appendChild(message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return p;
}

async function send() {
    const prompt = inputEl.value.trim();
    if (!prompt || sendButton.disabled) return;

    inputEl.value = "";
    addMessage("user", prompt);
    history.push({ role: "user", content: prompt });

    sendButton.disabled = true;
    const reply = addMessage("assistant", "…");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: history }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed.");
        reply.textContent = data.message;
        history.push({ role: "assistant", content: data.message });
    } catch (err) {
        reply.textContent = "⚠ " + err.message;
        history.pop(); // drop the unanswered prompt so the user can retry
    } finally {
        sendButton.disabled = false;
        inputEl.focus();
    }
}

sendButton.addEventListener("click", send);

inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        send();
    }
});

document.getElementById("clear-button").addEventListener("click", () => {
    history.length = 0;
    messagesEl.replaceChildren();
    inputEl.focus();
});

document.getElementById("sidebar-toggle").addEventListener("click", () => {
    document.querySelector(".conversations").classList.toggle("shown");
});
