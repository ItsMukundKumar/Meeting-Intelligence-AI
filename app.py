import warnings
warnings.filterwarnings("ignore")

from transformers.utils import logging
logging.set_verbosity_error()
import streamlit as st
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

load_dotenv()

# ── Page Config ─────────────────────────────────────

st.set_page_config(
    page_title="Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #111111;
    color: #F5F5F5;
    font-family: 'Sora', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #262626;
}

.stButton button {
    border-radius: 10px !important;
}

.main-title {
    text-align:center;
    padding: 1rem 0 2rem 0;
}

.main-title h1 {
    font-size: 2.3rem;
    margin-bottom: 0;
}

.main-title p {
    color: #888;
}

.block-card {
    background:#1B1B1B;
    border:1px solid #2B2B2B;
    padding:1.2rem;
    border-radius:14px;
    margin-bottom:1rem;
}

.small-title {
    color:#888;
    font-size:0.8rem;
    text-transform:uppercase;
    letter-spacing:1px;
    margin-bottom:0.6rem;
}

</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────

if "meetings" not in st.session_state:
    st.session_state.meetings = []

if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = None


# ── Pipeline ────────────────────────────────────────

def run_pipeline(source: str, language: str):

    from utils.audio_processor import process_input
    from core.transcriber import transcribe
    from core.summarize import generate_title, summarize
    from core.extractor import extract_items
    from core.rag_engine import build_rag_chain

    chunks = process_input(
        source=source,
        language=language
    )

    transcript = transcribe(
        chunks=chunks,
        language=language
    )

    summary = summarize(transcript=transcript)

    title = generate_title(summary=summary)

    items = extract_items(transcript=transcript)

    rag_chain = build_rag_chain(transcript=transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": items["action_items"],
        "decisions": items["decisions"],
        "open_questions": items["open_questions"],
        "rag_chain": rag_chain,
    }


# ── Sidebar ─────────────────────────────────────────

with st.sidebar:

    st.markdown("## 🎙️ Meetings")

    if st.button("➕ New Meeting", use_container_width=True):
        st.session_state.current_meeting_id = None
        st.rerun()

    st.divider()

    if len(st.session_state.meetings) == 0:

        st.caption("No meetings analysed yet.")

    else:

        for meeting in reversed(st.session_state.meetings):

            label = f"📌 {meeting['title']}"

            if st.button(
                label,
                key=meeting["id"],
                use_container_width=True
            ):
                st.session_state.current_meeting_id = meeting["id"]
                st.rerun()


# ── Header ──────────────────────────────────────────

st.markdown("""
<div class="main-title">
    <h1>🎙️ Meeting Intelligence</h1>
    <p>Transcribe · Summarise · Extract · Chat</p>
</div>
""", unsafe_allow_html=True)


# ── Upload Screen ───────────────────────────────────

if st.session_state.current_meeting_id is None:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown('<div class="block-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="small-title">Source</div>',
            unsafe_allow_html=True
        )

        source_type = st.radio(
            "Source",
            ["YouTube URL", "Upload File"],
            horizontal=True
        )

        source = None

        if source_type == "YouTube URL":

            source = st.text_input(
                "URL",
                placeholder="Paste YouTube URL...",
                label_visibility="collapsed"
            )

        else:

            uploaded_file = st.file_uploader(
                "Upload File",
                type=[
                    "mp3",
                    "wav",
                    "mp4",
                    "mkv",
                    "mpeg",
                    "webm"
                ],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:

                os.makedirs("temp_uploads", exist_ok=True)

                file_path = os.path.join(
                    "temp_uploads",
                    uploaded_file.name
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                source = file_path

                st.success(
                    f"Uploaded: {uploaded_file.name}"
                )

        language = st.selectbox(
            "Language",
            ["en", "hi"],
            format_func=lambda x:
            "🇬🇧 English"
            if x == "en"
            else "🇮🇳 Hindi / Hinglish"
        )

        process_btn = st.button(
            "Analyse Meeting →",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Process ───────────────────────────────

        if process_btn and source:

            progress = st.progress(0)

            status = st.empty()

            try:

                steps = [
                    "Loading Audio...",
                    "Transcribing...",
                    "Summarising...",
                    "Extracting Insights...",
                    "Building RAG..."
                ]

                for i, step in enumerate(steps):

                    status.info(step)

                    progress.progress((i + 1) * 20)

                result = run_pipeline(
                    source=source,
                    language=language
                )

                meeting_id = str(uuid.uuid4())

                meeting_data = {
                    "id": meeting_id,
                    "title": result["title"],
                    "summary": result["summary"],
                    "transcript": result["transcript"],
                    "action_items": result["action_items"],
                    "decisions": result["decisions"],
                    "open_questions": result["open_questions"],
                    "rag_chain": result["rag_chain"],
                    "chat_history": [],
                    "created_at": datetime.now().strftime(
                        "%d %b %Y • %I:%M %p"
                    )
                }

                st.session_state.meetings.append(
                    meeting_data
                )

                st.session_state.current_meeting_id = (
                    meeting_id
                )

                progress.progress(100)

                status.success("Analysis Complete")

                st.rerun()

            except Exception as e:

                st.error(f"Error: {e}")

        elif process_btn and not source:

            st.warning(
                "Please provide a source."
            )

# ── Meeting View ────────────────────────────────────

else:

    current_meeting = next(
        (
            m for m in st.session_state.meetings
            if m["id"] == st.session_state.current_meeting_id
        ),
        None
    )

    if current_meeting is None:

        st.error("Meeting not found")

    else:

        # ── Title ────────────────────────────────

        col1, col2 = st.columns([6, 1])

        with col1:

            st.markdown(
                f"""
                ## 📌 {current_meeting['title']}
                """
            )

            st.caption(
                current_meeting["created_at"]
            )

        with col2:

            if st.button(
                "← Back",
                use_container_width=True
            ):
                st.session_state.current_meeting_id = None
                st.rerun()

        st.divider()

        # ── Tabs ─────────────────────────────────

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary",
            "✅ Action Items",
            "🔑 Decisions",
            "❓ Questions",
            "💬 Chat"
        ])

        # ── Summary ─────────────────────────────

        with tab1:

            col1, col2 = st.columns([3, 2])

            with col1:

                st.markdown("""
                <div class="block-card">
                <div class="small-title">
                Summary
                </div>
                """, unsafe_allow_html=True)

                st.write(current_meeting["summary"])

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown("""
                <div class="block-card">
                <div class="small-title">
                Transcript Preview
                </div>
                """, unsafe_allow_html=True)

                st.code(
                    current_meeting["transcript"][:1200]
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                with st.expander(
                    "View Full Transcript"
                ):

                    st.text_area(
                        "Transcript",
                        value=current_meeting[
                            "transcript"
                        ],
                        height=400,
                        label_visibility="collapsed"
                    )

        # ── Action Items ────────────────────────

        with tab2:

            if not current_meeting["action_items"]:

                st.info(
                    "No action items found."
                )

            else:

                for item in current_meeting[
                    "action_items"
                ]:

                    st.markdown(
                        f"- {item}"
                    )

        # ── Decisions ───────────────────────────

        with tab3:

            if not current_meeting["decisions"]:

                st.info(
                    "No decisions found."
                )

            else:

                for item in current_meeting[
                    "decisions"
                ]:

                    st.markdown(
                        f"- {item}"
                    )

        # ── Questions ───────────────────────────

        with tab4:

            if not current_meeting[
                "open_questions"
            ]:

                st.info(
                    "No open questions found."
                )

            else:

                for item in current_meeting[
                    "open_questions"
                ]:

                    st.markdown(
                        f"- {item}"
                    )

        # ── Chat ────────────────────────────────

        with tab5:

            st.markdown(
                "### Ask Questions About Meeting"
            )

            # ── Existing Messages ──────────────

            for msg in current_meeting[
                "chat_history"
            ]:

                with st.chat_message(msg["role"]):

                    st.markdown(msg["content"])

            # ── Input ─────────────────────────

            question = st.chat_input(
                "Ask about this meeting..."
            )

            if question:

                current_meeting[
                    "chat_history"
                ].append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):

                    st.markdown(question)

                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):

                        answer = current_meeting[
                            "rag_chain"
                        ].invoke(question)

                        st.markdown(answer)

                current_meeting[
                    "chat_history"
                ].append({
                    "role": "assistant",
                    "content": answer
                })

                st.rerun()