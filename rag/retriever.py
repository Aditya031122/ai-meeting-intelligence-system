from __future__ import annotations

from typing import List, Optional, Tuple

import faiss
import numpy as np

from rag.embeddings import create_embeddings, create_faiss_index, model


_index: Optional[faiss.IndexFlatL2] = None
_text_chunks: List[str] = []


def load_embeddings(text_chunks: Optional[List[str]] = None) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Create (or recreate) a FAISS index from meeting transcript chunks.

    Parameters
    ----------
    text_chunks:
        Optional list of transcript chunks. If omitted, the function will
        reuse any chunks previously provided in this module.

    Returns
    -------
    (index, text_chunks)
        The FAISS index and the list of chunks it was built from.
    """
    global _index, _text_chunks

    if text_chunks is not None:
        _text_chunks = list(text_chunks)

    if not _text_chunks:
        raise ValueError("No transcript chunks available. Provide `text_chunks` to load_embeddings().")

    embeddings = create_embeddings(_text_chunks)
    _index = create_faiss_index(embeddings)

    return _index, _text_chunks


def retrieve(query: str, k: int = 3) -> List[str]:
    """
    Retrieve the top-k most relevant meeting transcript chunks for a query.

    Parameters
    ----------
    query:
        User question to search against the stored transcript chunks.
    k:
        Number of most relevant results to return.

    Returns
    -------
    list[str]
        The top-k relevant transcript chunks.
    """
    if _index is None or not _text_chunks:
        raise ValueError("Embeddings not loaded. Call load_embeddings(text_chunks) first.")

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = _index.search(query_embedding, k)

    results: List[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(_text_chunks):
            results.append(_text_chunks[idx])

    return results


if __name__ == "__main__":
    # Example usage with sample meeting transcripts
    sample_transcripts = [
        "We discussed the project roadmap for Q3 and aligned on milestones.",
        "Priya will prepare the deployment checklist for the backend services.",
        "Alex is responsible for the frontend UI updates and design tweaks.",
        "Rahul will handle backend deployment to production on Friday.",
        "The team agreed that security testing must be completed before go-live.",
    ]

    print("Creating FAISS index over sample meeting transcripts...")
    load_embeddings(sample_transcripts)

    user_question = "Who is responsible for backend deployment?"
    print(f"\nUser question: {user_question}")

    top_results = retrieve(user_question, k=3)

    print("\nTop retrieved transcript chunks:")
    for i, chunk in enumerate(top_results, start=1):
        print(f"{i}. {chunk}")

