from __future__ import annotations

import sys
from pathlib import Path
import uuid
from typing import Any, Dict, List

# Allow running as: `python backend/pipeline.py`
# (When executed this way, Python adds `backend/` to sys.path, not the project root.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.action_items import extract_action_items
from backend.analytics.sentiment import analyze_sentiment
from backend.analytics.speakers import detect_speakers
from backend.analytics.topics import detect_topics
from backend.storage import save_action_items, save_summary, save_transcript
from backend.summarizer import generate_summary
from rag.embeddings import create_embeddings


def _simple_chunk(text: str, *, max_len: int = 400) -> List[str]:
    """
    Very small, dependency-free chunker.
    Splits by sentence boundaries (.) and groups into ~max_len char chunks.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for s in sentences:
        # +1 for the period we add back
        s_len = len(s) + 1
        if buf and (buf_len + s_len) > max_len:
            chunks.append(". ".join(buf).strip() + ".")
            buf = []
            buf_len = 0
        buf.append(s)
        buf_len += s_len

    if buf:
        chunks.append(". ".join(buf).strip() + ".")

    return chunks or [text.strip()]


def process_meeting(transcript: str) -> Dict[str, Any]:
    """
    Process a meeting transcript:

    1) Generate summary + key points (via backend/summarizer.py)
    2) Extract action items (via backend/action_items.py)
    3) Save transcript/summary/action items as JSON (via backend/storage.py)
    4) Generate embeddings for transcript chunks (via rag/embeddings.py)

    Returns:
        {
          "summary": "...",
          "key_points": [...],
          "action_items": [...]
        }
    """
    if not transcript or not transcript.strip():
        return {"summary": "", "key_points": [], "action_items": []}

    meeting_id = str(uuid.uuid4())

    summary_result = generate_summary(transcript)
    action_result = extract_action_items(transcript)

    speakers_result = detect_speakers(transcript)
    topics_result = detect_topics(transcript, top_n=5)
    sentiment_result = analyze_sentiment(transcript)

    metadata = {
        "speakers": speakers_result.get("speakers", []),
        "speaker_detection": {
            "has_speaker_labels": speakers_result.get("has_speaker_labels", False),
        },
        "topics": topics_result,
        "sentiment": {
            "overall_label": sentiment_result.get("overall_label"),
            "overall_score": sentiment_result.get("overall_score"),
            "available": sentiment_result.get("available", False),
            "model": sentiment_result.get("model"),
        },
    }

    # Persist outputs (same meeting_id to keep them linked)
    save_transcript(transcript, meeting_id=meeting_id, metadata=metadata)
    save_summary(summary_result, meeting_id=meeting_id)
    save_action_items(action_result, meeting_id=meeting_id)

    # Generate embeddings (kept simple: compute and discard for now)
    # In a full system, you'd store these in FAISS and persist the index.
    # Prefer speaker-aware chunks (utterances) if available.
    utterances = speakers_result.get("utterances") or []
    if isinstance(utterances, list) and utterances and isinstance(utterances[0], dict):
        chunks = [f"{u.get('speaker', 'Unknown')}: {u.get('text', '')}".strip() for u in utterances]
        chunks = [c for c in chunks if len(c) > 5]
        if not chunks:
            chunks = _simple_chunk(transcript)
    else:
        chunks = _simple_chunk(transcript)
    _ = create_embeddings(chunks)

    return {
        "summary": summary_result.get("summary", ""),
        "key_points": summary_result.get("key_points", []),
        "action_items": action_result.get("action_items", []),
        "topics": topics_result.get("topics", []),
        "sentiment": metadata["sentiment"],
        "speakers": metadata["speakers"],
    }


if __name__ == "__main__":
    sample_transcript = """
    We discussed the drone delivery project timeline and the next milestones.
    Alex will handle UI implementation and coordinate with design.
    Rahul will deploy the backend by Friday and verify monitoring.
    We agreed to finalize the scope by Wednesday.
    """

    result = process_meeting(sample_transcript)
    print("Pipeline output:")
    print(result)

