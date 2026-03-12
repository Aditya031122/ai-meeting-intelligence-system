from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def detect_topics(transcript: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Best-effort topic detection for a meeting transcript.

    Uses KeyBERT if available (higher quality). Otherwise falls back to a simple
    keyword frequency approach.

    Returns:
      {
        "topics": ["backend deployment", "ui implementation", ...],
        "keywords": ["deployment", "backend", ...],
        "method": "keybert" | "frequency"
      }
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {"topics": [], "keywords": [], "method": "frequency"}

    keybert_topics = _try_keybert(transcript, top_n=top_n)
    if keybert_topics is not None:
        topics, keywords = keybert_topics
        return {"topics": topics, "keywords": keywords, "method": "keybert"}

    keywords = _top_keywords_frequency(transcript, top_n=max(top_n * 2, 8))
    # Create simple multiword "topics" by pairing adjacent frequent keywords if present.
    topics = _compose_topics(transcript, keywords, top_n=top_n)
    return {"topics": topics, "keywords": keywords[:top_n], "method": "frequency"}


def _try_keybert(text: str, top_n: int) -> Tuple[List[str], List[str]] | None:
    try:
        from keybert import KeyBERT

        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=max(top_n * 2, 8),
        )
        phrases = [p for p, _ in kws]
        # Favor 2-grams as "topics", but keep 1-grams if needed.
        topics = [p for p in phrases if len(p.split()) >= 2][:top_n]
        if len(topics) < top_n:
            topics.extend([p for p in phrases if len(p.split()) == 1][: max(0, top_n - len(topics))])
        keywords = [p for p in phrases if len(p.split()) == 1][: max(top_n, 5)]
        return topics, keywords
    except Exception:
        return None


def _top_keywords_frequency(text: str, top_n: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    stop = _stopwords()
    filtered = [t for t in tokens if t not in stop]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(top_n)]


def _compose_topics(text: str, keywords: List[str], top_n: int) -> List[str]:
    # Naive phrase detection: find bigrams that appear in text using frequent keywords.
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    bigrams = list(zip(words, words[1:]))
    kw_set = set(keywords[: max(top_n * 2, 10)])
    bigram_counts = Counter([" ".join(bg) for bg in bigrams if bg[0] in kw_set and bg[1] in kw_set])
    topics = [p for p, _ in bigram_counts.most_common(top_n)]
    # If no good bigrams found, just return top keywords as topics.
    if not topics:
        topics = keywords[:top_n]
    return topics


def _stopwords() -> set[str]:
    return {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "was",
        "were",
        "are",
        "is",
        "to",
        "of",
        "in",
        "on",
        "we",
        "i",
        "you",
        "a",
        "an",
        "it",
        "as",
        "be",
        "by",
        "or",
        "from",
        "at",
        "our",
        "their",
        "they",
        "will",
        "should",
        "can",
        "could",
        "need",
        "needs",
        "needed",
        "about",
        "discussed",
        "meeting",
        "today",
        "yesterday",
        "tomorrow",
    }


if __name__ == "__main__":
    sample = "Alex will handle UI implementation. Rahul will deploy backend services on Friday. We discussed monitoring and rollout."
    print(detect_topics(sample))

