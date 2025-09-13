RAG-based Medical FAQ Chatbot
Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for answering medical FAQs.
It uses:

SentenceTransformers (all-MiniLM-L6-v2) for embeddings

ChromaDB for vector storage & retrieval

Groq API for fast LLM-powered response generation

FastAPI to serve the chatbot backend

Streamlit to provide a simple chat-based UI

⚙️ Tech Stack

Python 3.9+

FastAPI
 – API framework

ChromaDB
 – vector database

SentenceTransformers
 – embeddings

Groq API
 – LLM inference

Streamlit
 – frontend UI

Uvicorn
 – ASGI server
Installation

Clone repo & create virtual environment

git clone <your-repo-url>
cd rag-medical-faq
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Set environment variables
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Usage
1. Preprocess dataset into ChromaDB

Ensure your dataset Medical_FAQ.csv has columns:

Question

Answer

Then run:

python preprocess.py


This will:

Read the CSV

Encode Q&A pairs into embeddings

Store them in a persistent ChromaDB collection (./chroma_persist/)

2. Start FastAPI backend
uvicorn app:app --reload


Backend runs at:
http://localhost:8000

Example request:

curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Who is at risk for Lymphocytic Choriomeningitis (LCM)?"}'


Response:

{
  "question": "Who is at risk for Lymphocytic Choriomeningitis (LCM)?",
  "answer": "Individuals of all ages who come into contact with urine, feces, saliva, or blood of wild mice..."
}

3. Run Streamlit frontend
streamlit run ui.py


Open browser: http://localhost:8501

You can now chat with the bot interactively.

Design Choices

Embeddings: Local SentenceTransformer (all-MiniLM-L6-v2) → avoids API cost & latency.

Vector DB: ChromaDB PersistentClient → ensures data survives restarts.

LLM: Groq (openai/gpt-oss-20b or llama-3.1-8b/70b) → fast inference.

Backend: FastAPI → clean, modular, easy to extend.

Frontend: Streamlit → quick, no-JS chat UI.
