import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import os

# --- Config ---
DATA_PATH = "Medical_FAQ.csv"   # CSV file with Question, Answer columns
CHROMA_PATH = "./chroma_persist"
COLLECTION_NAME = "medical_faqs"

# --- Setup ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def build_collection():
    df = pd.read_csv(DATA_PATH)

    if not {"Question", "Answer"}.issubset(df.columns):
        raise ValueError("CSV must have 'Question' and 'Answer' columns")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row["Question"]).strip()
        answer = str(row["Answer"]).strip()

        if not question or not answer:
            continue  # skip empty rows

        combined_text = f"Q: {question}\nA: {answer}"

        embedding = embedding_model.encode(combined_text).tolist()

        try:
            collection.add(
                documents=[combined_text],
                embeddings=[embedding],
                ids=[f"faq_{idx}"],  # safer ID prefix
            )
        except Exception as e:
            print(f"⚠️ Skipping row {idx} due to error: {e}")

    print(f"✅ Collection '{COLLECTION_NAME}' built and persisted at {CHROMA_PATH}")

if __name__ == "__main__":
    build_collection()
