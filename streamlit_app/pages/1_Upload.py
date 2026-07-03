import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Upload · Agentic RAG",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
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
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    color: #ececec !important;
    text-align: left !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] button:hover {
    background: #2a2a2a !important;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }

.block-container {
    max-width: 700px !important;
    margin: auto !important;
    padding-top: 2.5rem !important;
}
.upload-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}
.upload-sub {
    color: #666;
    font-size: 0.82rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 8px 10px;display:flex;align-items:center;gap:10px;'>
        <span style='font-size:1.5rem;'>🧠</span>
        <span style='font-size:1.05rem;font-weight:700;color:#fff;'>Agentic RAG</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.page_link("app.py",            label="💬  Chat")
    st.page_link("pages/1_Upload.py", label="📂  Upload Documents")

    st.divider()
    st.markdown(
        "<p style='font-size:0.72rem;color:#444;padding:4px 8px;line-height:1.6;'>"
        "Upload PDF, DOCX, or TXT files. Files are chunked, embedded and "
        "indexed into the RAG pipeline automatically.</p>",
        unsafe_allow_html=True,
    )

# ── Main ───────────────────────────────────────────────────────────
st.markdown('<h2 class="upload-title">📂 Upload Documents</h2>', unsafe_allow_html=True)
st.markdown(
    '<div class="upload-sub">Admin Panel · PDF, DOCX, TXT supported</div>',
    unsafe_allow_html=True,
)

st.divider()

source = st.text_input(
    "Source Label",
    placeholder="e.g. HR Policy, Product Manual, Q3 Report",
    help="A short label to identify where this document comes from.",
)

uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("⬆️  Upload", use_container_width=True, type="primary"):

    if not source.strip():
        st.warning("Please enter a source label.")
    elif not uploaded_files:
        st.warning("Please select at least one file.")
    else:
        progress = st.progress(0, text="Uploading...")

        for i, f in enumerate(uploaded_files):
            try:
                r = requests.post(
                    f"{API_BASE}/documents/upload",
                    data={"source": source.strip()},
                    files={"file": (f.name, f.getvalue(), f.type)},
                    timeout=120,
                )
                r.raise_for_status()
                d    = r.json()
                warn = " ⚠️ Stale cleanup failed." if d.get("stale_version_cleanup_failed") else ""
                st.success(
                    f"✅ **{f.name}** · Version {d.get('version','?')} · "
                    f"ID: `{d.get('document_id','?')}`{warn}"
                )
            except requests.exceptions.Timeout:
                st.error(f"❌ **{f.name}** — timed out.")
            except Exception as e:
                st.error(f"❌ **{f.name}** — {e}")

            progress.progress(
                int((i + 1) / len(uploaded_files) * 100),
                text=f"Uploading {i+1}/{len(uploaded_files)}...",
            )

        progress.empty()
