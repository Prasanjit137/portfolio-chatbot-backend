import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "my_knowledge"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs locally

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for i in range(end, max(start, end-100), -1):
                if text[i] in (' ', '\n', '.', '!', '?'):
                    end = i + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def main():
    # 1. Load and chunk documents
    all_chunks = []
    for filename in os.listdir(KNOWLEDGE_DIR):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(KNOWLEDGE_DIR, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        print(f"{filename}: {len(chunks)} chunks")
    print(f"Total chunks: {len(all_chunks)}")

    # 2. Create embeddings using local sentence‑transformer
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Computing embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Save to disk
    os.makedirs("rag_data", exist_ok=True)
    with open("rag_data/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    np.save("rag_data/embeddings.npy", embeddings)
    print("✅ Saved chunks and embeddings to rag_data/")

if __name__ == "__main__":
    main()