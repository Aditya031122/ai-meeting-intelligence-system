from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st


# Allow running via: `streamlit run ui/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.meeting_qa import answer_question  # noqa: E402
from backend.pipeline import process_meeting  # noqa: E402
from backend.speech_to_text import transcribe_audio  # noqa: E402
from backend.storage import load_all_transcripts  # noqa: E402
from ui.dashboard import render_dashboard  # noqa: E402


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_uploaded_audio(uploaded_file) -> Path:
    audio_dir = _ensure_dir(PROJECT_ROOT / "data" / "audio")
    file_id = str(uuid.uuid4())
    safe_name = Path(uploaded_file.name).name
    dest = audio_dir / f"{file_id}_{safe_name}"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


st.set_page_config(page_title="AI Meeting Intelligence System", layout="centered")
st.title("AI Meeting Intelligence System")

page = st.sidebar.radio("Navigation", options=["Upload & Process", "Meeting Dashboard"])
if page == "Meeting Dashboard":
    render_dashboard()
    st.stop()

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary_result" not in st.session_state:
    st.session_state.summary_result = {"summary": "", "key_points": [], "action_items": []}
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = ""


st.header("Upload Audio")
uploaded = st.file_uploader(
    "Upload a meeting audio file",
    type=["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac"],
)

col1, col2 = st.columns([1, 1])
with col1:
    do_transcribe = st.button("Transcribe", disabled=uploaded is None)
with col2:
    do_process = st.button("Generate Summary + Action Items", disabled=not bool(st.session_state.transcript.strip()))

if do_transcribe and uploaded is not None:
    try:
        audio_path = _save_uploaded_audio(uploaded)
        st.session_state.last_audio_path = str(audio_path)
        with st.spinner("Transcribing audio with Whisper (may take a while on first run)..."):
            st.session_state.transcript = transcribe_audio(str(audio_path))
        st.success("Transcription complete.")
    except Exception as exc:
        st.error(f"Transcription failed: {exc}")


st.header("Transcript")
if st.session_state.last_audio_path:
    st.caption(f"Latest audio: `{st.session_state.last_audio_path}`")
st.text_area("Transcript text", value=st.session_state.transcript, height=220)


st.header("Summary")
if do_process:
    try:
        with st.spinner("Generating summary and extracting action items..."):
            st.session_state.summary_result = process_meeting(st.session_state.transcript)
        st.success("Processing complete (saved to `data/`).")
    except Exception as exc:
        st.error(f"Processing failed: {exc}")

summary_text = st.session_state.summary_result.get("summary", "")
st.write(summary_text if summary_text else "_No summary yet._")

key_points = st.session_state.summary_result.get("key_points", []) or []
if key_points:
    st.subheader("Key Discussion Points")
    for p in key_points:
        st.write(f"- {p}")


st.header("Action Items")
action_items = st.session_state.summary_result.get("action_items", []) or []
if not action_items:
    st.write("_No action items yet._")
else:
    for item in action_items:
        person = item.get("person", "Unknown")
        task = item.get("task", "")
        deadline = item.get("deadline", "Not specified")
        st.write(f"- **{person}**: {task} _(deadline: {deadline})_")

st.header("Topics & Sentiment")
topics = st.session_state.summary_result.get("topics", []) or []
sentiment = st.session_state.summary_result.get("sentiment", {}) or {}
speakers = st.session_state.summary_result.get("speakers", []) or []

if speakers:
    st.write(f"**Speakers:** {', '.join([str(s) for s in speakers])}")
if topics:
    st.write(f"**Topics:** {', '.join([str(t) for t in topics])}")
if sentiment:
    st.write(
        f"**Sentiment:** {sentiment.get('overall_label', 'NEUTRAL')} "
        f"(score: {sentiment.get('overall_score', 0.0):.2f})"
    )


st.header("Ask Questions")
question = st.text_input("Ask a question about your past meetings", placeholder="Who is responsible for backend deployment?")
ask = st.button("Answer Question", disabled=not bool(question.strip()))

if ask:
    try:
        # Use all stored transcripts; include current transcript as well (if present).
        records = load_all_transcripts()
        transcripts = [r.get("transcript", "") for r in records if r.get("transcript")]
        if st.session_state.transcript.strip():
            transcripts.append(st.session_state.transcript.strip())

        with st.spinner("Retrieving relevant context and generating an answer..."):
            answer = answer_question(question.strip(), transcripts)
        st.write(answer)
    except Exception as exc:
        st.error(f"Q&A failed: {exc}")


st.divider()
st.caption(
    "Example questions: "
    "`Who is responsible for UI implementation?` · "
    "`What was the meeting about?`"
)

