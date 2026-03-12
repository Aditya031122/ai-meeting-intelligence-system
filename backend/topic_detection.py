from sklearn.feature_extraction.text import TfidfVectorizer


def extract_topics(transcript: str, top_k=5):
    """
    Extract important keywords/topics from a meeting transcript
    using TF-IDF.
    """

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform([transcript])

    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    topics = [word for word, score in sorted_scores[:top_k]]

    return topics


if __name__ == "__main__":

    transcript = """
    We discussed the drone delivery project timeline.
    Alex will implement the UI for the drone control dashboard.
    Rahul will deploy backend services for drone communication.
    The team also reviewed testing plans for drone navigation.
    """

    topics = extract_topics(transcript)

    print("Detected Topics:")
    print(topics)