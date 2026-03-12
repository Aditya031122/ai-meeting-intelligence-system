from transformers import pipeline


sentiment_model = pipeline("sentiment-analysis")


def analyze_sentiment(transcript: str):

    sentences = [s.strip() for s in transcript.split(".") if s.strip()]

    results = []

    for sentence in sentences:
        sentiment = sentiment_model(sentence)[0]

        results.append({
            "text": sentence,
            "sentiment": sentiment["label"],
            "score": sentiment["score"]
        })

    # simple overall sentiment
    positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
    negative = sum(1 for r in results if r["sentiment"] == "NEGATIVE")

    overall = "POSITIVE" if positive >= negative else "NEGATIVE"

    return {
        "overall_sentiment": overall,
        "sentence_sentiments": results
    }


if __name__ == "__main__":

    transcript = """
    The drone project is progressing well.
    The backend deployment faced some delays.
    The UI design looks excellent.
    """

    result = analyze_sentiment(transcript)

    print(result)