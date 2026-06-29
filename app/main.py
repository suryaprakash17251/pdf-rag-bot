# FastAPI routes
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ingest import build_and_save_index
from retriever import search
from generator import generate_answer
app = FastAPI(title="PDF RAG Bot")

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "PDF RAG Bot is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # save uploaded file to data/
    pdf_path = f"data/{file.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # build FAISS index from uploaded PDF
    build_and_save_index(pdf_path)

    return {"message": f"PDF '{file.filename}' uploaded and indexed successfully"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # check index exists
    if not os.path.exists("vector_store/index.faiss"):
        return {"error": "No PDF uploaded yet. Please upload a PDF first."}

    # retrieve + generate
    chunks = search(request.question)
    answer = generate_answer(request.question, chunks)

    return {
        "question": request.question,
        "answer": answer,
        "sources": chunks
    }