from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Utterance:
    speaker: str
    text: str
    start_char: int
    end_char: int


_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?P<speaker>(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)|(?:SPEAKER\s*\d+)|(?:HOST)|(?:MODERATOR))\s*:\s*(?P<text>.+?)\s*$",
    flags=re.IGNORECASE,
)


def detect_speakers(transcript: str) -> Dict[str, Any]:
    """
    Best-effort speaker detection from a transcript.

    This is **text-only** parsing (not audio diarization). It works when the transcript
    already contains speaker labels such as:
      - "Alex: let's ship it"
      - "Speaker 1: I agree"
      - "HOST: welcome everyone"

    Returns:
      {
        "speakers": ["Alex", "Rahul", ...],
        "utterances": [
          {"speaker": "Alex", "text": "...", "start_char": 12, "end_char": 55},
          ...
        ],
        "has_speaker_labels": true/false
      }
    """
    if not transcript or not transcript.strip():
        return {"speakers": [], "utterances": [], "has_speaker_labels": False}

    utterances: List[Utterance] = []
    has_labels = False

    # Keep character offsets by scanning line-by-line and tracking running index.
    lines = transcript.splitlines() or [transcript]
    cursor = 0

    for line in lines:
        line_len = len(line)
        match = _SPEAKER_LINE_RE.match(line)
        if match:
            has_labels = True
            speaker_raw = match.group("speaker").strip()
            text = match.group("text").strip()
            speaker = _normalize_speaker(speaker_raw)

            # Approximate offsets: locate text segment within the full transcript.
            start = cursor + max(line.lower().find(text.lower()), 0)
            end = start + len(text)
            utterances.append(Utterance(speaker=speaker, text=text, start_char=start, end_char=end))

        cursor += line_len + 1  # +1 for newline

    if not utterances:
        # Fallback: single-speaker transcript
        t = transcript.strip()
        utterances = [Utterance(speaker="Unknown", text=t, start_char=0, end_char=len(t))]

    speakers = sorted({u.speaker for u in utterances})

    return {
        "speakers": speakers,
        "utterances": [
            {"speaker": u.speaker, "text": u.text, "start_char": u.start_char, "end_char": u.end_char}
            for u in utterances
        ],
        "has_speaker_labels": has_labels,
    }


def _normalize_speaker(s: str) -> str:
    s_clean = re.sub(r"\s+", " ", s.strip())
    s_up = s_clean.upper()
    if s_up.startswith("SPEAKER"):
        # Keep canonical format: "Speaker 1"
        num = re.findall(r"\d+", s_clean)
        return f"Speaker {num[0]}" if num else "Speaker"
    if s_up in {"HOST", "MODERATOR"}:
        return s_up.title()
    # Title-case person names
    return " ".join([p[:1].upper() + p[1:].lower() for p in s_clean.split(" ") if p])


if __name__ == "__main__":
    sample = """HOST: Welcome everyone.\nAlex: I can take the UI.\nRahul: I'll deploy backend Friday.\n"""
    print(detect_speakers(sample))

