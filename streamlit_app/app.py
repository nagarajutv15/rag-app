import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Force black sidebar everywhere ── */
[data-testid="stSidebar"] {
    background: #171717 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #171717 !important;
}
[data-testid="stSidebarContent"] {
    background: #171717 !important;
}
[data-testid="stSidebar"] * {
    color: #ececec !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2e2e2e !important;
}

/* Hide default Streamlit chrome & nav */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    color: #ececec !important;
    text-align: left !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] button:hover {
    background: #2a2a2a !important;
}

/* New Chat button */
[data-testid="stSidebar"] button[kind="secondary"]:first-of-type {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    font-weight: 600 !important;
}

/* 3-dot popup card */
.dot-menu {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 4px 0;
    margin: 0 8px 6px 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
.dot-menu button {
    width: 100%;
    text-align: left !important;
    padding: 8px 14px !important;
    border-radius: 6px !important;
    color: #ff6b6b !important;
    font-size: 0.85rem !important;
}
.dot-menu button:hover {
    background: #3a2a2a !important;
}

/* Main content */
.block-container {
    padding-top: 2rem !important;
    max-width: 800px !important;
    margin: auto !important;
}

/* White title */
.main-title {
    font-size: 1.6rem;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
}
.main-sub {
    text-align: center;
    color: #666;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── State ──────────────────────────────────────────────────────────
if "session_id"    not in st.session_state: st.session_state.session_id    = None
if "messages"      not in st.session_state: st.session_state.messages      = []
if "sessions"      not in st.session_state: st.session_state.sessions      = []
if "chat_counter"  not in st.session_state: st.session_state.chat_counter  = 0
if "open_menu"     not in st.session_state: st.session_state.open_menu     = None

# ── Helpers ────────────────────────────────────────────────────────
def new_chat():
    st.session_state.session_id = None
    st.session_state.messages   = []
    st.session_state.open_menu  = None

def load_history(sid):
    try:
        r = requests.get(f"{API_BASE}/chat/history/{sid}", timeout=10)
        r.raise_for_status()
        data = r.json()
        st.session_state.messages   = data.get("messages", [])
        st.session_state.session_id = sid
        st.session_state.open_menu  = None
    except Exception as e:
        st.sidebar.error(f"Failed to load: {e}")

def delete_session(sid):
    try:
        requests.delete(f"{API_BASE}/chat/history/{sid}", timeout=10)
    except Exception:
        pass
    st.session_state.sessions = [s for s in st.session_state.sessions if s["id"] != sid]
    if st.session_state.session_id == sid:
        new_chat()
    st.session_state.open_menu = None

def send_message(question):
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"question": question, "session_id": st.session_state.session_id},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        sid  = data["session_id"]
        if st.session_state.session_id is None:
            st.session_state.session_id = sid
            st.session_state.chat_counter += 1
            # use first ~4 words of question as label
            words = question.strip().split()
            label = " ".join(words[:4]) + ("..." if len(words) > 4 else "")
            st.session_state.sessions.append({
                "id":    sid,
                "label": label,
            })
        return data["answer"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {e}"

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style='padding:18px 8px 10px;display:flex;align-items:center;gap:10px;'>
        <span style='font-size:1.5rem;'>🧠</span>
        <span style='font-size:1.05rem;font-weight:700;color:#fff;'>Agentic RAG</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()

    st.divider()

    st.markdown(
        "<p style='font-size:0.7rem;color:#555;padding:0 4px 6px;"
        "letter-spacing:.06em;'>CHATS</p>",
        unsafe_allow_html=True,
    )

    # ── Session list ──
    for s in reversed(st.session_state.sessions):
        sid      = s["id"]
        label    = s["label"]
        is_open  = st.session_state.open_menu == sid
        is_active = st.session_state.session_id == sid

        col_label, col_dots = st.columns([6, 1])

        with col_label:
            btn_label = f"{'▶  ' if is_active else '💬  '}{label}"
            if st.button(btn_label, key=f"load_{sid}", use_container_width=True):
                if not is_active:
                    load_history(sid)
                else:
                    st.session_state.open_menu = None
                st.rerun()

        with col_dots:
            if st.button("⋯", key=f"dots_{sid}"):
                st.session_state.open_menu = sid if not is_open else None
                st.rerun()

        # ── Popup delete card ──
        if is_open:
            st.markdown('<div class="dot-menu">', unsafe_allow_html=True)
            if st.button("🗑  Delete Chat", key=f"del_{sid}", use_container_width=True):
                delete_session(sid)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.sessions:
        st.markdown(
            "<p style='font-size:0.8rem;color:#444;padding:4px 8px;'>"
            "No chats yet.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.page_link("pages/1_Upload.py", label="📂  Upload Documents")

# ── Main Chat ──────────────────────────────────────────────────────
st.markdown('<h2 class="main-title">🧠 Agentic RAG</h2>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">Your intelligent document assistant — ask anything, get precise answers.</div>',
    unsafe_allow_html=True,
)

if st.session_state.session_id:
    st.caption(f"Session · `{st.session_state.session_id}`")

st.divider()

if not st.session_state.messages:
    st.markdown(
        "<br><br><p style='text-align:center;color:#555;font-size:0.95rem;'>"
        "Send a message to start a new chat.</p>",
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = send_message(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
