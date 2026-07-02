import streamlit as st
import requests
import sseclient

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------
# Custom CSS — clean Claude-like UI
# ------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* hide default streamlit header/footer */
        #MainMenu, footer, header { visibility: hidden; }

        /* slim sidebar */
        [data-testid="stSidebar"] {
            background-color: #f7f7f8;
            border-right: 1px solid #e5e5e5;
        }

        /* chat bubbles */
        .user-msg {
            background: #f0f0f0;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 6px 0;
            max-width: 80%;
            margin-left: auto;
            text-align: right;
        }
        .assistant-msg {
            background: #ffffff;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 6px 0;
            max-width: 80%;
            border: 1px solid #e5e5e5;
        }

        /* center title */
        .title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 600;
            padding: 1rem 0 0.5rem;
            color: #1a1a1a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session State Init
# ------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sessions" not in st.session_state:
    st.session_state.sessions = []   # list of session_ids created this run

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def new_chat():
    """Reset to a blank state — session_id created on first message."""
    st.session_state.session_id = None
    st.session_state.messages = []


def load_history(session_id: str):
    """Fetch chat history from the API and load into session state."""
    try:
        r = requests.get(
            f"{API_BASE}/chat/history/{session_id}",
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        st.session_state.messages = data.get("messages", [])
        st.session_state.session_id = session_id
    except Exception as e:
        st.sidebar.error(f"Failed to load history: {e}")


def send_message(question: str):
    """
    POST to /chat (non-streaming).
    On the first message session_id is None — the backend creates one
    and returns it; we persist it for all subsequent messages.
    """
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
        }
        r = requests.post(
            f"{API_BASE}/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        # persist session_id returned by backend
        session_id = data["session_id"]
        if st.session_state.session_id is None:
            st.session_state.session_id = session_id
            # track for sidebar
            if session_id not in st.session_state.sessions:
                st.session_state.sessions.append(session_id)

        return data["answer"]

    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {e}"

# ------------------------------------------------------------------
# Sidebar — New Chat + session history
# ------------------------------------------------------------------

with st.sidebar:

    st.markdown("### 🤖 Agentic RAG")
    st.divider()

    if st.button("➕  New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.markdown("**Recent Sessions**")

    if st.session_state.sessions:
        for sid in reversed(st.session_state.sessions):
            label = f"💬 ...{sid[-8:]}"
            if st.button(label, key=sid, use_container_width=True):
                load_history(sid)
                st.rerun()
    else:
        st.caption("No sessions yet.")

# ------------------------------------------------------------------
# Main — chat area
# ------------------------------------------------------------------

st.markdown('<div class="title">Agentic RAG Assistant</div>', unsafe_allow_html=True)

# show session id subtly when active
if st.session_state.session_id:
    st.caption(f"Session: `{st.session_state.session_id}`")

st.divider()

# render chat history
chat_container = st.container()

with chat_container:

    if not st.session_state.messages:
        st.markdown(
            "<br><br><p style='text-align:center; color:#aaa;'>Send a message to get started.</p>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------

if prompt := st.chat_input("Ask anything..."):

    # optimistically add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = send_message(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()
