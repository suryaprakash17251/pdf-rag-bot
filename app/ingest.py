# PDF extraction + chunking + embedding
import fitz  # PyMuPDF
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def extract_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) 
              for i in range(0, len(words), chunk_size)]
    return chunks

def build_and_save_index(pdf_path):
    import faiss
    chunks = extract_chunks(pdf_path)
    print(f"Total chunks: {len(chunks)}")

    embeddings = MODEL.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/index.faiss")
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Index saved to vector_store/")
    return index, chunks

if __name__ == "__main__":
    build_and_save_index("data/sample.pdf")