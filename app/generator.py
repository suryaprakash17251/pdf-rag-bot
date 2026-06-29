# Groq LLM call
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    sample_chunks = [
        "CatBoost achieved 97% accuracy with a macro AUC of 0.9973.",
        "SHAP analysis showed C-reactive protein and troponin are the most influential features.",
        "The framework integrates RAG to generate patient-specific explanations."
    ]
    answer = generate_answer("What accuracy did the model achieve?", sample_chunks)
    print("\nAnswer:", answer)