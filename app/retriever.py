# FAISS search
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    index = faiss.read_index("vector_store/index.faiss")
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query, k=3):
    index, chunks = load_index()
    query_embedding = MODEL.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    _, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    results = search("what is this document about?")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---\n{chunk}")