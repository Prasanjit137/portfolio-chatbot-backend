# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
import numpy as np
from collections import defaultdict

OLLAMA_API_URL = "https://Prasanjit137-ollama-api.hf.space"
DEFAULT_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "nomic-embed-text"

# Check safely if embeddings have been built yet
CHUNKS_PATH = "rag_data/chunks.pkl"
EMBEDDINGS_PATH = "rag_data/embeddings.npy"

chunks = []
embeddings = None

if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")
else:
    print("⚠️ Warning: Vector data files not found. Run ingestion or push to documents/ first.")

def get_query_embedding(text):
    resp = requests.post(f"{OLLAMA_API_URL}/api/embeddings",
                         json={"model": EMBEDDING_MODEL, "prompt": text},
                         timeout=30)
    if resp.status_code != 200:
        resp = requests.post(f"{OLLAMA_API_URL}/api/embeddings",
                             json={"model": DEFAULT_MODEL, "prompt": text},
                             timeout=30)
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return np.dot(a, b) / denom

def retrieve_context(query, k=3):
    global chunks, embeddings
    if not chunks or embeddings is None:
        return ""
        
    q_emb = get_query_embedding(query)
    similarities = []
    for i, doc_emb in enumerate(embeddings):
        sim = cosine_similarity(q_emb, doc_emb)
        similarities.append((sim, i))
        
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_indices = [idx for _, idx in similarities[:k]]
    return "\n\n".join([chunks[i] for i in top_indices])

SYSTEM_PROMPT = """You are Prasanjit Sarkar, an AI engineer and full‑stack developer. 
You have 2+ years of experience architecting autonomous agentic workflows and scalable GenAI systems. 
Your expertise includes multi‑agent systems, RAG, LangChain, LLM fine‑tuning, and full‑stack integration.
You have built over 10 AI projects, including a wearable safety device and an Ollama‑powered chatbot.
Answer as Prasanjit – helpful, friendly, professional. Use the provided context if relevant."""

conversations = defaultdict(list)
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_input = data.get('chatInput')
    session_id = data.get('sessionId', 'default')
    if not user_input:
        return jsonify({"error": "Missing chatInput"}), 400

    history = conversations[session_id]

    try:
        context = retrieve_context(user_input)
    except Exception as e:
        print(f"Retrieval error: {e}")
        context = ""

    system_msg = SYSTEM_PROMPT
    if context:
        system_msg += f"\n\nRelevant information from my documents:\n{context}\n\nUse this to answer accurately."
    
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    payload = {"model": DEFAULT_MODEL, "messages": messages, "stream": False}
    try:
        resp = requests.post(f"{OLLAMA_API_URL}/api/chat", json=payload, timeout=90)
        resp.raise_for_status()
        reply = resp.json().get("message", {}).get("content", "")
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        
        if len(history) > 20:
            conversations[session_id] = history[-20:]
            
        return jsonify({"output": reply})
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"output": "Error handling request. Please try again."}), 500

@app.route('/clear', methods=['POST'])
def clear():
    data = request.get_json() or {}
    sess = data.get('sessionId', 'default')
    if sess in conversations:
        del conversations[sess]
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)