from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def analyze_sentiment(transcript: str, *, chunk_size: int = 350) -> Dict[str, Any]:
    """
    Analyze sentiment for a meeting transcript (best-effort).

    Uses a HuggingFace transformers pipeline if available:
      model: `distilbert-base-uncased-finetuned-sst-2-english`

    Returns:
      {
        "overall_label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
        "overall_score": float,
        "chunks": [{"text": "...", "label": "...", "score": 0.98}, ...],
        "model": "<model name or None>",
        "available": true/false
      }
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {
            "overall_label": "NEUTRAL",
            "overall_score": 0.0,
            "chunks": [],
            "model": None,
            "available": False,
        }

    chunks = _chunk_text(transcript, chunk_size=chunk_size)

    pipe, model_name = _try_get_pipeline()
    if pipe is None:
        # Fallback: neutral sentiment without external deps
        return {
            "overall_label": "NEUTRAL",
            "overall_score": 0.0,
            "chunks": [],
            "model": None,
            "available": False,
        }

    results = pipe(chunks)
    chunk_results: List[Dict[str, Any]] = []
    for text, r in zip(chunks, results):
        chunk_results.append(
            {
                "text": text,
                "label": r.get("label", "NEUTRAL"),
                "score": float(r.get("score", 0.0)),
            }
        )

    overall_label, overall_score = _aggregate(chunk_results)

    return {
        "overall_label": overall_label,
        "overall_score": overall_score,
        "chunks": chunk_results,
        "model": model_name,
        "available": True,
    }


def _try_get_pipeline() -> Tuple[Optional[object], Optional[str]]:
    try:
        from transformers import pipeline

        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        pipe = pipeline("sentiment-analysis", model=model_name)
        return pipe, model_name
    except Exception:
        return None, None


def _chunk_text(text: str, *, chunk_size: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for w in words:
        w_len = len(w) + 1
        if buf and (buf_len + w_len) > chunk_size:
            chunks.append(" ".join(buf).strip())
            buf = []
            buf_len = 0
        buf.append(w)
        buf_len += w_len
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


def _aggregate(chunk_results: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not chunk_results:
        return "NEUTRAL", 0.0

    # Map POS/NEG to signed score; average across chunks.
    signed_scores: List[float] = []
    for r in chunk_results:
        label = str(r.get("label", "")).upper()
        score = float(r.get("score", 0.0))
        if label.startswith("POS"):
            signed_scores.append(score)
        elif label.startswith("NEG"):
            signed_scores.append(-score)
        else:
            signed_scores.append(0.0)

    avg = sum(signed_scores) / max(len(signed_scores), 1)
    if avg > 0.15:
        return "POSITIVE", float(avg)
    if avg < -0.15:
        return "NEGATIVE", float(avg)
    return "NEUTRAL", float(avg)


if __name__ == "__main__":
    sample = "Great progress today. Some concerns about deadlines, but overall we are confident."
    print(analyze_sentiment(sample))

