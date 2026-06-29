from ingest import build_and_save_index
from retriever import search
from generator import generate_answer
import os

def run_pipeline(pdf_path, question):
    # Step 1 — build index if not exists
    if not os.path.exists("vector_store/index.faiss"):
        print("Building index...")
        build_and_save_index(pdf_path)

    # Step 2 — retrieve relevant chunks
    print("Searching for relevant chunks...")
    chunks = search(question)

    # Step 3 — generate answer
    print("Generating answer...\n")
    answer = generate_answer(question, chunks)

    return answer

if __name__ == "__main__":
    questions = [
        "What accuracy did the model achieve?",
        "Which biomarkers are most important?",
        "What is the conclusion of this paper?"
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {run_pipeline('data/sample.pdf', q)}")
        print("-" * 50)