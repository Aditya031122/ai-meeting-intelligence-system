from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_root() -> Path:
    # backend/ is one level under the repo root
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "data"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(folder: Path, payload: Dict[str, Any], file_stem: Optional[str] = None) -> Path:
    folder = _ensure_dir(folder)
    stem = file_stem or payload.get("id") or str(uuid.uuid4())
    path = folder / f"{stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_transcript(
    transcript_text: str,
    *,
    meeting_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a transcript to `data/transcripts/` as JSON.

    Returns the path to the saved JSON file.
    """
    payload: Dict[str, Any] = {
        "id": meeting_id or str(uuid.uuid4()),
        "created_at": _now_iso(),
        "transcript": transcript_text,
        "metadata": metadata or {},
    }
    return _write_json(_data_dir() / "transcripts", payload, file_stem=payload["id"])


def save_summary(
    summary: Dict[str, Any] | str,
    *,
    meeting_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a summary to `data/summaries/` as JSON.

    `summary` can be a string or a dict (e.g. {"summary": "...", "key_points": [...]}).
    """
    payload: Dict[str, Any] = {
        "id": meeting_id or str(uuid.uuid4()),
        "created_at": _now_iso(),
        "summary": summary,
        "metadata": metadata or {},
    }
    return _write_json(_data_dir() / "summaries", payload, file_stem=payload["id"])


def save_action_items(
    action_items: List[Any] | Dict[str, Any] | str,
    *,
    meeting_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save action items to `data/actions/` as JSON.

    `action_items` can be a list, dict, or string.
    """
    payload: Dict[str, Any] = {
        "id": meeting_id or str(uuid.uuid4()),
        "created_at": _now_iso(),
        "action_items": action_items,
        "metadata": metadata or {},
    }
    return _write_json(_data_dir() / "actions", payload, file_stem=payload["id"])


def load_all_transcripts() -> List[Dict[str, Any]]:
    """
    Load all transcript JSON files from `data/transcripts/`.

    Returns a list of transcript records.
    """
    folder = _ensure_dir(_data_dir() / "transcripts")
    records: List[Dict[str, Any]] = []

    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("_path", str(path))
                records.append(data)
        except Exception:
            # Skip unreadable/corrupt files
            continue

    return records


def load_meeting(meeting_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a single meeting by id, merging transcript + summary + action items if present.

    Returns:
      {
        "id": ...,
        "transcript": {...} | None,
        "summary": {...} | None,
        "actions": {...} | None
      }
    """
    meeting_id = (meeting_id or "").strip()
    if not meeting_id:
        return None

    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    base = _data_dir()
    transcript = _read_json(base / "transcripts" / f"{meeting_id}.json")
    summary = _read_json(base / "summaries" / f"{meeting_id}.json")
    actions = _read_json(base / "actions" / f"{meeting_id}.json")

    if transcript is None and summary is None and actions is None:
        return None

    return {"id": meeting_id, "transcript": transcript, "summary": summary, "actions": actions}


def load_all_meetings() -> List[Dict[str, Any]]:
    """
    Load all meetings, merging transcript + summary + actions by meeting id.

    This is useful for dashboards and search.
    """
    base = _data_dir()
    transcripts_dir = _ensure_dir(base / "transcripts")
    summaries_dir = _ensure_dir(base / "summaries")
    actions_dir = _ensure_dir(base / "actions")

    ids = set([p.stem for p in transcripts_dir.glob("*.json")])
    ids |= set([p.stem for p in summaries_dir.glob("*.json")])
    ids |= set([p.stem for p in actions_dir.glob("*.json")])

    meetings: List[Dict[str, Any]] = []
    for meeting_id in sorted(ids):
        m = load_meeting(meeting_id)
        if m is not None:
            meetings.append(m)
    return meetings

