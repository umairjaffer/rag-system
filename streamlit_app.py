import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be called first
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind — PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (Light / Clean UI)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main background */
        .stApp {
            background-color: #ffffff;
            color: #111827;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f9fafb;
            border-right: 1px solid #e5e7eb;
        }

        /* Cards */
        .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }

        /* Question bubble */
        .q-bubble {
            background: #4f46e5;
            color: #ffffff;
            border-radius: 18px 18px 4px 18px;
            padding: 0.75rem 1.1rem;
            margin-left: auto;
            max-width: 80%;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Answer bubble */
        .a-bubble {
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 18px 18px 18px 4px;
            padding: 0.85rem 1.1rem;
            max-width: 85%;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #111827;
        }

        /* Source chip */
        .source-chip {
            display: inline-block;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            color: #4f46e5;
            border-radius: 20px;
            padding: 2px 12px;
            font-size: 0.75rem;
            margin: 2px;
        }

        /* Step styles */
        .step-done  { color: #10b981; font-size: 0.88rem; }
        .step-active{ color: #4f46e5; font-size: 0.88rem; font-weight: 600; }
        .step-wait  { color: #9ca3af; font-size: 0.88rem; }

        /* Hide Streamlit default UI */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        /* Input styling */
        .stTextInput input {
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
            color: #111827 !important;
            border-radius: 10px !important;
        }

        .stTextInput input:focus {
            border-color: #4f46e5 !important;
            box-shadow: 0 0 0 2px rgba(79,70,229,0.15) !important;
        }

        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []   # stores chat history

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []  # already processed PDFs

# ---------------------------------------------------------------------------
# Load RAG engine (cached once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine():
    """Load and cache the RAG engine (embedding model loads only once)."""
    from app.rag_engine import RAGEngine
    return RAGEngine()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 DocMind")
    st.markdown(
        "<span style='color:#6b7280; font-size:0.82rem;'>"
        "PDF Question Answering — Groq + FAISS + HuggingFace"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # System status section
    st.markdown("#### System Status")
    status_slot = st.empty()

    with status_slot.container():
        with st.spinner("Loading embedding model..."):
            try:
                engine = load_engine()
                st.session_state.rag_engine = engine
            except Exception as e:
                st.error(f"Failed to load engine: {e}")
                st.stop()

    status_slot.empty()

    col1, col2 = st.columns(2)
    col1.metric("Status", "Ready")
    col2.metric("Index", "Loaded" if engine.is_ready else "Empty")

    st.divider()

    # File uploader
    st.markdown("#### Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        already_indexed = uploaded_file.name in st.session_state.indexed_files

        if already_indexed:
            st.success(f"✓ {uploaded_file.name} already indexed")
        else:
            st.markdown(f"**Processing:** `{uploaded_file.name}`")
            progress_bar = st.progress(0)

            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            step4 = st.empty()
            result_slot = st.empty()

            # Step 1: Save file
            step1.markdown("<div class='step-active'>Reading PDF...</div>", unsafe_allow_html=True)
            progress_bar.progress(10)

            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            save_path = upload_dir / uploaded_file.name

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            step1.markdown("<div class='step-done'>Saved successfully</div>", unsafe_allow_html=True)
            progress_bar.progress(25)

            # Step 2: Extract
            step2.markdown("<div class='step-active'>Extracting pages...</div>", unsafe_allow_html=True)
            progress_bar.progress(40)

            step2.markdown("<div class='step-done'>Pages extracted</div>", unsafe_allow_html=True)
            progress_bar.progress(55)

            # Step 3: Embedding
            step3.markdown("<div class='step-active'>Embedding chunks...</div>", unsafe_allow_html=True)
            progress_bar.progress(65)

            try:
                chunks = engine.index_pdf(
                    file_path=str(save_path),
                    filename=uploaded_file.name,
                )

                step3.markdown(
                    f"<div class='step-done'>{chunks} chunks embedded</div>",
                    unsafe_allow_html=True,
                )
                progress_bar.progress(85)

                # Step 4: Save index
                step4.markdown("<div class='step-active'>Saving FAISS index...</div>", unsafe_allow_html=True)
                progress_bar.progress(100)

                step4.markdown("<div class='step-done'>Index saved</div>", unsafe_allow_html=True)

                st.session_state.indexed_files.append(uploaded_file.name)
                result_slot.success("PDF ready for Q&A!")

            except Exception as e:
                step3.error(f"Error: {e}")
                progress_bar.empty()

    st.divider()

    # Indexed files
    if st.session_state.indexed_files:
        st.markdown("#### Indexed Documents")
        for fname in st.session_state.indexed_files:
            st.markdown(f"<div class='source-chip'>📄 {fname}</div>", unsafe_allow_html=True)

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # How it works
    with st.expander("How it works"):
        st.markdown(
            """
            1. Upload PDF → saved locally  
            2. Extract text using PyPDF  
            3. Split into chunks  
            4. Generate embeddings (HuggingFace)  
            5. Store in FAISS index  
            6. Ask questions → retrieve top chunks → LLM answer  
            """
        )

# ---------------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------------
st.markdown("### Ask Questions")

engine_ready = (
    st.session_state.rag_engine is not None
    and st.session_state.rag_engine.is_ready
)

if not engine_ready and not st.session_state.indexed_files:
    st.info("Upload a PDF from the sidebar to get started 👈")

# Display chat history
for msg in st.session_state.messages:
    col1, col2 = st.columns([1, 4])
    with col2:
        st.markdown(f"<div class='q-bubble'>{msg['question']}</div>", unsafe_allow_html=True)

    col3, col4 = st.columns([4, 1])
    with col3:
        st.markdown(f"<div class='a-bubble'>{msg['answer']}</div>", unsafe_allow_html=True)

    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for src in msg["sources"]:
                st.markdown(f"📄 {src.source_file} | Page {src.page}")
                st.code(src.chunk_text)

# ---------------------------------------------------------------------------
# Input box
# ---------------------------------------------------------------------------
st.divider()

with st.form("question_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])

    with col1:
        user_question = st.text_input(
            "Question",
            placeholder="Ask something about your document...",
            label_visibility="collapsed",
            disabled=not engine_ready,
        )

    with col2:
        submitted = st.form_submit_button(
            "Ask",
            use_container_width=True,
            disabled=not engine_ready,
            type="primary",
        )

if submitted and user_question.strip():
    question = user_question.strip()

    with st.spinner("Generating answer..."):
        try:
            result = st.session_state.rag_engine.query(question=question)

            st.session_state.messages.append({
                "question": result.question,
                "answer": result.answer,
                "sources": result.sources,
            })

        except Exception as e:
            st.error(f"Error: {e}")

    st.rerun()