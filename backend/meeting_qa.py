from __future__ import annotations

from typing import List

from transformers import pipeline

from rag.retriever import load_embeddings, retrieve


_generator = None


def _get_generator():
    """
    Lazily load a small HuggingFace text-generation model (gpt2).
    """
    global _generator
    if _generator is None:
        _generator = pipeline(
            task="text-generation",
            model="gpt2",
            max_new_tokens=128,
        )
    return _generator


def answer_question(query: str, transcripts: List[str]) -> str:
    """
    Answer a question about meetings using retrieved transcript chunks.

    Workflow:
        query -> retrieve relevant transcripts -> construct prompt -> generate answer

    Args:
        query: User question about the meetings.
        transcripts: List of full/partial meeting transcripts (chunks).

    Returns:
        Generated answer as a string.
    """
    if not transcripts:
        return "No transcripts available to answer this question."

    # Build / refresh FAISS index over the provided transcripts
    load_embeddings(transcripts)

    # Retrieve relevant context for the query
    relevant_chunks = retrieve(query, k=3)

    context = "\n".join(relevant_chunks)

    prompt = (
        "You are an assistant answering questions about meeting transcripts.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    generator = _get_generator()
    outputs = generator(prompt, num_return_sequences=1)
    full_text = outputs[0]["generated_text"]

    # Extract only the part after "Answer:" for a cleaner response
    answer = full_text.split("Answer:", 1)[-1].strip()
    return answer


if __name__ == "__main__":
    # Example usage
    example_transcripts = [
        "We discussed the drone delivery project timeline and main milestones.",
        "Alex is responsible for UI implementation and frontend integration.",
        "Rahul will handle backend deployment and infrastructure setup.",
        "The meeting was primarily about aligning on responsibilities and timelines.",
    ]

    questions = [
        "Who is responsible for UI implementation?",
        "What was the meeting about?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        ans = answer_question(q, example_transcripts)
        print(f"Answer: {ans}")

