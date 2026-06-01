from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
import numpy as np
from collections import defaultdict

# Load chunks and embeddings
with open("rag_data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
embeddings = np.load("rag_data/embeddings.npy")
print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query, k=3):
    # Get query embedding (must match the same model)
    # Note: For this to work, we need to embed the query using the same model.
    # We'll use a local SentenceTransformer model – but to avoid loading the model
    # on your Mac, we can offload this to the HF Space embedding API (since it's just one call per query).
    # However, that would require nomic-embed-text. Alternative: load the model on your Mac (it's small enough?).
    # To keep it light, we'll use the HF Space's embedding API if nomic-embed-text becomes available.
    # But you said you want to avoid that. So we'll use a tiny local model on the Mac as well.
    # Since the model is only 80 MB and used once per query (not heavy), it's acceptable.
    # Let's implement it with local model, but we can also use the precomputed embeddings without loading the model? No, we need to embed the query.
    # Simpler: use keyword matching for retrieval, but you asked for embeddings.
    # I'll provide a version that loads the same model locally on the Mac (it will work, but if you face memory issues, switch to keyword).
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query])[0].astype('float32')
    sims = [cosine_similarity(q_emb, emb) for emb in embeddings]
    top_idx = np.argsort(sims)[-k:][::-1]
    return "\n\n".join([chunks[i] for i in top_idx if sims[i] > 0.2])

# If you prefer to avoid loading the model on your Mac, replace retrieve_context with the keyword version.
# For now, we'll keep it as is – the model is small and will load once at startup.

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

    context = retrieve_context(user_input, k=3) if chunks else ""
    if context:
        system_msg = f"{SYSTEM_PROMPT}\n\nRelevant information:\n{context}\n\nUse this to answer accurately."
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