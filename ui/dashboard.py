from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from backend.storage import load_all_meetings


def render_dashboard() -> None:
    st.header("Meeting Search Dashboard")

    meetings = load_all_meetings()
    if not meetings:
        st.info("No meetings found yet. Process a meeting first to populate `data/`.")
        return

    # Flatten a small searchable view
    rows: List[Dict[str, Any]] = []
    for m in meetings:
        t = m.get("transcript") or {}
        s = m.get("summary") or {}
        a = m.get("actions") or {}
        meta = (t.get("metadata") or {}) if isinstance(t, dict) else {}

        created_at = ""
        if isinstance(t, dict):
            created_at = str(t.get("created_at", ""))

        rows.append(
            {
                "id": m.get("id", ""),
                "created_at": created_at,
                "topics": _safe_topics(meta),
                "speakers": _safe_speakers(meta),
                "sentiment": _safe_sentiment_label(meta),
                "summary": _safe_summary_text(s),
                "action_items_count": _safe_actions_count(a),
                "transcript": t.get("transcript", "") if isinstance(t, dict) else "",
            }
        )

    # Filters
    st.subheader("Filters")
    q = st.text_input("Search (summary/transcript)", placeholder="backend deployment, UI, roadmap...")
    speaker_filter = st.text_input("Speaker contains", placeholder="Alex")
    topic_filter = st.text_input("Topic contains", placeholder="deployment")
    sentiment_filter = st.selectbox("Sentiment", options=["Any", "POSITIVE", "NEUTRAL", "NEGATIVE"])

    filtered = []
    for r in rows:
        if sentiment_filter != "Any" and r["sentiment"] != sentiment_filter:
            continue
        if speaker_filter.strip() and speaker_filter.lower() not in ", ".join(r["speakers"]).lower():
            continue
        if topic_filter.strip() and topic_filter.lower() not in ", ".join(r["topics"]).lower():
            continue
        if q.strip():
            blob = f"{r.get('summary','')}\n{r.get('transcript','')}".lower()
            if q.lower() not in blob:
                continue
        filtered.append(r)

    st.subheader(f"Meetings ({len(filtered)}/{len(rows)})")

    meeting_labels = [_label_for_row(r) for r in filtered]
    selected_label = st.selectbox("Select a meeting", options=meeting_labels)
    selected = _row_by_label(filtered, selected_label)

    if not selected:
        st.warning("No meeting selected.")
        return

    st.divider()
    st.subheader("Meeting Details")
    st.caption(f"Meeting ID: `{selected['id']}`")
    st.write(f"**Created:** {selected.get('created_at') or 'Unknown'}")
    st.write(f"**Speakers:** {', '.join(selected['speakers']) if selected['speakers'] else 'Unknown'}")
    st.write(f"**Topics:** {', '.join(selected['topics']) if selected['topics'] else '—'}")
    st.write(f"**Sentiment:** {selected.get('sentiment') or '—'}")
    st.write(f"**Action items:** {selected.get('action_items_count', 0)}")

    st.subheader("Summary")
    st.write(selected.get("summary") or "_No summary stored._")

    with st.expander("Transcript"):
        st.text_area("Transcript", value=selected.get("transcript", ""), height=240)


def _safe_topics(meta: Dict[str, Any]) -> List[str]:
    t = meta.get("topics") if isinstance(meta, dict) else None
    if isinstance(t, dict):
        topics = t.get("topics", [])
        return [str(x) for x in topics if str(x).strip()]
    return []


def _safe_speakers(meta: Dict[str, Any]) -> List[str]:
    speakers = meta.get("speakers") if isinstance(meta, dict) else None
    if isinstance(speakers, list):
        return [str(x) for x in speakers if str(x).strip()]
    return []


def _safe_sentiment_label(meta: Dict[str, Any]) -> str:
    s = meta.get("sentiment") if isinstance(meta, dict) else None
    if isinstance(s, dict):
        return str(s.get("overall_label") or "NEUTRAL").upper()
    return "NEUTRAL"


def _safe_summary_text(summary_record: Dict[str, Any]) -> str:
    if not isinstance(summary_record, dict):
        return ""
    val = summary_record.get("summary")
    if isinstance(val, dict):
        return str(val.get("summary", "")).strip()
    return str(val or "").strip()


def _safe_actions_count(actions_record: Dict[str, Any]) -> int:
    if not isinstance(actions_record, dict):
        return 0
    items = actions_record.get("action_items")
    if isinstance(items, dict):
        items = items.get("action_items", [])
    if isinstance(items, list):
        return len(items)
    return 0


def _label_for_row(r: Dict[str, Any]) -> str:
    created = r.get("created_at") or ""
    created_short = created
    try:
        created_short = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    topic = (r.get("topics") or ["Meeting"])[0] if isinstance(r.get("topics"), list) else "Meeting"
    return f"{created_short} | {topic} | {r.get('id','')}"


def _row_by_label(rows: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    for r in rows:
        if _label_for_row(r) == label:
            return r
    return None

