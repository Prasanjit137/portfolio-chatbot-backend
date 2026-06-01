from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open("rag_data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("rag_data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
embeddings = np.load("rag_data/tfidf_embeddings.npy")
print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")

def retrieve_context(query, k=3):
    # Transform query into the same TF‑IDF space
    query_vec = vectorizer.transform([query]).toarray()
    # Compute cosine similarity
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[-k:][::-1]
    # Only return if similarity > 0.1 (avoid random matches)
    results = [chunks[i] for i in top_indices if sims[i] > 0.1]
    return "\n\n".join(results) if results else ""

SYSTEM_PROMPT = """You are Prasanjit Sarkar, an AI engineer and full‑stack developer. 
You have 2+ years of experience architecting autonomous agentic workflows and scalable GenAI systems. 
Your expertise includes multi‑agent systems, RAG, LangChain, LLM fine‑tuning, and full‑stack integration.
You have built over 10 AI projects, including a wearable safety device and an Ollama‑powered chatbot.
Answer as Prasanjit – helpful, friendly, professional. Use the provided context if relevant."""

conversations = defaultdict(list)
app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "https://Prasanjit137-ollama-api.hf.space"
CHAT_MODEL = "llama3.2:1b"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('chatInput')
    session_id = data.get('sessionId', 'default')
    if not user_input:
        return jsonify({"error": "Missing chatInput"}), 400

    context = retrieve_context(user_input, k=3)
    if context:
        system_msg = f"{SYSTEM_PROMPT}\n\nRelevant information from my documents:\n{context}\n\nUse this to answer accurately."
    else:
        system_msg = SYSTEM_PROMPT

    history = conversations[session_id]
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    try:
        resp = requests.post(f"{OLLAMA_API_URL}/api/chat",
                             json={"model": CHAT_MODEL, "messages": messages, "stream": False},
                             timeout=60)
        resp.raise_for_status()
        reply = resp.json().get("message", {}).get("content", "")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 20:
            conversations[session_id] = history[-20:]
        return jsonify({"output": reply})
    except Exception as e:
        print(e)
        return jsonify({"output": "Error. Please try again."}), 500

@app.route('/clear', methods=['POST'])
def clear():
    sess = request.get_json().get('sessionId', 'default')
    if sess in conversations:
        del conversations[sess]
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)