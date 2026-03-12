from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_action_items(transcript: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract action items from a meeting transcript.

    Prefer an NLP-based approach (spaCy) when available; otherwise fall back to a
    lightweight regex extractor.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {"action_items": []}

    items = _extract_with_spacy(transcript)
    if items is None:
        items = _extract_with_regex(transcript)

    return {"action_items": items}


def _extract_with_spacy(transcript: str) -> Optional[List[Dict[str, str]]]:
    """
    spaCy-based extraction (best-effort).

    Detects patterns like:
      - "<PERSON> will <VERB> <...>"
      - "<PERSON> should <VERB> <...>"
      - "<PERSON> needs to <VERB> <...>"

    Returns None if spaCy (or its model) is unavailable.
    """
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    nlp = _load_spacy_model(spacy)
    if nlp is None:
        return None

    doc = nlp(transcript)
    items: List[Dict[str, str]] = []

    obligation_markers = {"will", "should", "need", "needs", "must", "plan", "plans"}
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Identify the first PERSON entity in the sentence as assignee candidate.
        person = None
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                person = ent.text.strip()
                break

        if not person:
            continue

        # Find an obligation marker token near the person.
        marker_idx = None
        for i, tok in enumerate(sent):
            if tok.lemma_.lower() in obligation_markers:
                marker_idx = i
                break

        if marker_idx is None:
            continue

        # Task text: everything after the marker token.
        task = sent[marker_idx + 1 :].text.strip()
        if not task:
            continue

        deadline = _extract_deadline_hint(sent_text)
        items.append({"person": person, "task": task, "deadline": deadline})

    return _dedupe_items(items)


def _load_spacy_model(spacy_module: Any):
    for name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy_module.load(name)
        except Exception:
            continue
    return None


def _extract_with_regex(transcript: str) -> List[Dict[str, str]]:
    """
    Lightweight fallback extractor.
    """
    action_items: List[Dict[str, str]] = []
    sentences = re.split(r"[.\n]+", transcript)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # patterns like: "Alex will handle UI implementation"
        match = re.search(r"(\b[A-Z][a-z]+)\s+(?:will|should|needs?\s+to|must)\s+(.*)", sentence)
        if not match:
            continue

        person = match.group(1)
        task = match.group(2).strip()
        if not task:
            continue

        deadline = _extract_deadline_hint(sentence)
        action_items.append({"person": person, "task": task, "deadline": deadline})

    return _dedupe_items(action_items)


def _extract_deadline_hint(text: str) -> str:
    # Very small deadline heuristic.
    for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        if day.lower() in text.lower():
            return day
    match = re.search(r"\bby\s+([A-Za-z0-9 ]{3,20})\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Not specified"


def _dedupe_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for it in items:
        key = (it.get("person", "").strip().lower(), it.get("task", "").strip().lower())
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


if __name__ == "__main__":

    transcript = """
    We discussed the drone delivery project timeline.
    Alex will handle UI implementation.
    Rahul will deploy the backend by Friday.
    """

    result = extract_action_items(transcript)

    print(result)
