from typing import Dict, List
from transformers import pipeline


def load_model():
    """
    Load a small text generation model for summarization.
    """
    generator = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=120
    )
    return generator


def generate_summary(transcript: str) -> Dict[str, List[str] | str]:

    if not transcript.strip():
        return {"summary": "", "key_points": []}

    model = load_model()

    prompt = f"""
Summarize the meeting and list key points.

Meeting Transcript:
{transcript}

Summary:
"""

    output = model(prompt)[0]["generated_text"]

    # extract summary portion
    summary = output.split("Summary:")[-1].strip()

    sentences = transcript.split(".")
    key_points = [s.strip() for s in sentences if len(s.strip()) > 10][:3]

    return {
        "summary": summary,
        "key_points": key_points
    }


if __name__ == "__main__":

    transcript = """
    We discussed the drone delivery project timeline.
    Alex will handle UI implementation.
    Rahul will deploy the backend by Friday.
    """

    result = generate_summary(transcript)

    print(result)
