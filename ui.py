import streamlit as st
import requests

# --- FastAPI backend URL ---
API_URL = "http://localhost:8000/query"  # adjust if deployed elsewhere

st.set_page_config(page_title="Medical FAQ Chatbot", page_icon="🩺")
st.title("🩺 Medical FAQ Chatbot (RAG + Groq)")

st.write("Ask me any medical question from the knowledge base.")

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_query = st.chat_input("Type your medical question here...")

if user_query:
    # Save user query
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Send request to FastAPI
    try:
        response = requests.post(API_URL, json={"question": user_query})
        response.raise_for_status()
        answer = response.json().get("answer", "⚠️ No answer found.")
    except Exception as e:
        answer = f"⚠️ Error contacting backend: {e}"

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Display conversation ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
