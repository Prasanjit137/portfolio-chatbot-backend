# ingest.py
import os
import glob
import pickle
import requests
import numpy as np

OLLAMA_API_URL = "https://Prasanjit137-ollama-api.hf.space"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_MODEL = "llama3.2:1b"

def get_embedding(text):
    try:
        resp = requests.post(
            f"{OLLAMA_API_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        if resp.status_code != 200:
            resp = requests.post(
                f"{OLLAMA_API_URL}/api/embeddings",
                json={"model": DEFAULT_MODEL, "prompt": text},
                timeout=60
            )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"Error generating embedding for text snippet: {text[:30]}... | Error: {e}")
        return None

def load_and_chunk_docs():
    chunks = []
    if not os.path.exists("documents"):
        os.makedirs("documents")
        
    for filepath in glob.glob("documents/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            # Splits text cleanly into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks.extend(paragraphs)
    return chunks

def main():
    print("Starting document ingestion pipeline via Hugging Face...")
    chunks = load_and_chunk_docs()
    
    if not chunks:
        print("ℹ️ No documents found in 'documents/'. Place .txt files there to build embeddings.")
        return

    print(f"Found {len(chunks)} text chunks. Generating vector matrix...")
    
    valid_chunks = []
    embeddings_list = []
    
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            valid_chunks.append(chunk)
            embeddings_list.append(emb)
            
    if not embeddings_list:
        print("❌ Ingestion aborted: No embeddings were returned from the remote API.")
        return

    os.makedirs("rag_data", exist_ok=True)

    with open("rag_data/chunks.pkl", "wb") as f:
        pickle.dump(valid_chunks, f)
        
    np.save("rag_data/embeddings.npy", np.array(embeddings_list, dtype=np.float32))
    print(f"Successfully serialized {len(valid_chunks)} chunks to rag_data/!")

if __name__ == "__main__":
    main()