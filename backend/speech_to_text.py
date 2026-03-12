from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import whisper


_MODEL: Optional[whisper.Whisper] = None


def _get_model() -> whisper.Whisper:
    """
    Lazily load and cache the Whisper model.
    """
    global _MODEL
    if _MODEL is None:
        model_name = os.getenv("WHISPER_MODEL", "base")
        _MODEL = whisper.load_model(model_name)
    return _MODEL


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file using OpenAI Whisper.

    - Accepts a path to an audio file.
    - Returns the full transcript text.
    - Saves the transcript as a .txt file inside the `data/transcripts` folder.

    Parameters
    ----------
    file_path:
        Path to the audio file to transcribe.

    Returns
    -------
    str
        The full transcript text.

    Raises
    ------
    FileNotFoundError
        If the provided audio file path does not exist.
    RuntimeError
        If transcription fails for any reason.
    """
    audio_path = Path(file_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    try:
        result = model.transcribe(str(audio_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    transcript_text: str = (result.get("text") or "").strip()

    # Determine transcripts directory: <project_root>/data/transcripts
    project_root = Path(__file__).resolve().parent
    transcripts_dir = project_root / "data" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    transcript_file = transcripts_dir / f"{audio_path.stem}.txt"
    transcript_file.write_text(transcript_text, encoding="utf-8")

    return transcript_text

