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

STOPWORDS = {
    'i', 'me', 'my', 'you', 'he', 'she', 'it', 'we', 'they', 'a', 'an', 'the',
    'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this', 'that',
    'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
    'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on',
    'with', 'without', 'be', 'been', 'was', 'were', 'are', 'am', 'do', 'does',
    'did', 'doing', 'have', 'has', 'having', 'not', 'no', 'yes', 'at', 'by',
    'after', 'before', 'up', 'down', 'into', 'onto', 'off', 'out', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'only', 'own', 'same', 'so', 'than', 'that', 'then', 'these', 'those',
    'very', 'just', 'but', 'do', 'does', 'did', 'doing', 'have', 'has',
    'having', 'don’t', 'cant', 'cannot', 'should', 'would', 'could', 'will',
    'shall', 'may', 'might', 'must'
}

def extract_keywords(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def retrieve_context(query, top_k=3):
    q_keywords = set(extract_keywords(query))
    if not q_keywords:
        return ""
    scored = []
    for chunk in chunks:
        chunk_keywords = set(extract_keywords(chunk))
        score = len(q_keywords & chunk_keywords)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    selected = [chunk for _, chunk in scored[:top_k]]
    return "\n\n".join(selected)

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

    context = retrieve_context(user_input)
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
                             timeout=90)
        resp.raise_for_status()
        reply = resp.json().get("message", {}).get("content", "")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 20:
            conversations[session_id] = history[-20:]
        return jsonify({"output": reply})
    except Exception as e:
        print(e)
        return jsonify({"output": "I'm having trouble. Please try again later."}), 500

@app.route('/clear', methods=['POST'])
def clear():
    sess = request.get_json().get('sessionId', 'default')
    if sess in conversations:
        del conversations[sess]
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)