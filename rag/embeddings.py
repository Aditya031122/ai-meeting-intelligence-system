from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_chunks):

    embeddings = model.encode(text_chunks)

    return embeddings


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def search(index, query, text_chunks, k=2):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k)

    results = [text_chunks[i] for i in indices[0]]

    return results


if __name__ == "__main__":

    transcripts = [
        "We discussed the drone delivery project timeline.",
        "Alex will handle UI implementation.",
        "Rahul will deploy backend by Friday."
    ]

    embeddings = create_embeddings(transcripts)

    index = create_faiss_index(embeddings)

    query = "Who is responsible for backend?"

    results = search(index, query, transcripts)

    print("Search Results:")
    print(results)
