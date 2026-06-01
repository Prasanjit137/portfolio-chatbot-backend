# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
import re
from collections import defaultdict

# Load chunks
with open("rag_data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"Loaded {len(chunks)} chunks")

def simple_retrieve(query, chunks, top_k=3):
    stopwords = {'i', 'me', 'my', 'you', 'he', 'she', 'it', 'we', 'they', 'a', 'an', 'the',
                 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this', 'that',
                 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
                 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on',
                 'with', 'without', 'be', 'been', 'was', 'were', 'are', 'am', 'do', 'does',
                 'did', 'doing', 'have', 'has', 'having', 'not', 'no', 'yes'}
    query_words = set(re.findall(r'\b[a-z]+\b', query.lower())) - stopwords
    if not query_words:
        return chunks[:top_k]
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\b[a-z]+\b', chunk.lower()))
        score = len(query_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:top_k] if _ > 0]

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

    relevant_chunks = simple_retrieve(user_input, chunks, top_k=3)
    context = "\n\n".join(relevant_chunks) if relevant_chunks else ""

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