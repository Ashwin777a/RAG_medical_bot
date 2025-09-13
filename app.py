from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import uvicorn
import os
from dotenv import load_dotenv
from groq import Groq

# --- Load environment ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=groq_api_key)

# --- FastAPI app ---
app = FastAPI(title="Medical FAQ Chatbot")

# --- Embedding model + Chroma ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client_chroma = chromadb.PersistentClient(path="./chroma_persist")
collection = client_chroma.get_collection(name="medical_faqs")


class QueryRequest(BaseModel):
    question: str


def get_embedding(text: str):
    """Generate embedding from local sentence-transformer"""
    return embedding_model.encode(text).tolist()


def generate_answer(context: str, query: str) -> str:
    """Use Groq to generate an answer from retrieved context"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical assistant. "
                "Answer strictly based on the provided context. "
                "If unsure, say you are unsure and suggest consulting a healthcare professional."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    # Call Groq Chat API
    chat_completion = client_groq.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=messages,
        temperature=0.0,
        max_tokens=300,
    )

    return chat_completion.choices[0].message.content.strip()


@app.post("/query")
def query_bot(request: QueryRequest):
    user_query = request.question
    query_embedding = get_embedding(user_query)

    results = collection.query(query_embeddings=[query_embedding], n_results=2)

    if not results["documents"] or not results["documents"][0]:
        return {"question": user_query, "answer": "No relevant context found."}

    # Flatten retrieved docs into one context block
    retrieved_context = "\n\n".join(doc for doc in results["documents"][0])
    print(f"Retrieved Context:\n{retrieved_context}\n")

    answer = generate_answer(retrieved_context, user_query)

    return {"question": user_query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
