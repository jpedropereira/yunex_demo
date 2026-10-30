"""Demo beat 1: the bare model, with only a system prompt.

Deliberately thin: the model call lives in ``yunex_demo.chat``.

Streamlit reruns this whole script top to bottom on every interaction (each
message sent, each button click). Anything that must survive between runs,
like the conversation so far, has to live in ``st.session_state``.
"""

import httpx
import streamlit as st
from ollama import ResponseError

from yunex_demo import chat
from yunex_demo.config import settings

# --- 1. Page header: say up front what this beat is (and is not) ------------
st.set_page_config(page_title="Bare chatbot", page_icon=":material/chat:", layout="wide")

st.title("Bare chatbot")
st.caption(
    f"`{settings.model_main}` with no tools, no retrieval, and no schema -- "
    "the baseline the next pages make reliable."
)

# Show the system prompt so the audience sees it is the *only* instruction
# the model gets.
with st.expander("System prompt"):
    st.code(chat.SYSTEM_PROMPT, language="text")

# --- 2. Conversation memory -------------------------------------------------
# The model itself is stateless: it remembers nothing between calls. The page
# keeps the history and resends all of it every turn, which is how the model
# "remembers" earlier messages.
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
history = st.session_state["chat_history"]

if st.button("Clear chat", icon=":material/delete:"):
    history.clear()

# --- 3. Replay the conversation so far ---------------------------------------
# Because of the rerun model, the chat window starts empty on every run, so
# earlier messages are redrawn from history before handling the new one.
# Model output is untrusted, so it is rendered as plain markdown only:
# Streamlit escapes raw HTML unless unsafe_allow_html is set, and it never is here.
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Handle a new question ------------------------------------------------
# chat_input returns the text on the run right after the user presses Enter,
# and None on every other run.
if prompt := st.chat_input("Ask anything"):
    # Record and show the user's message first, so it appears immediately.
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 5. Ask the model and stream the answer ------------------------------
    with st.chat_message("assistant"):
        try:
            # write_stream renders fragments as they arrive and returns the
            # full text once the model finishes.
            reply = st.write_stream(chat.stream_reply(history))
        # The ollama client turns "server unreachable" into a builtin ConnectionError.
        except (ConnectionError, httpx.HTTPError, ResponseError) as exc:
            # Drop the unanswered question so a retry does not send it twice.
            history.pop()
            st.error(f"Model call failed: {exc}. Check the Stack Health page.")
        else:
            # Save the answer so the next turn includes it as context.
            history.append({"role": "assistant", "content": reply})
