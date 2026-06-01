import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KNOWLEDGE_DIR = "my_knowledge"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

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

    # Create TF‑IDF vectors
    print("Creating TF‑IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_chunks)   # sparse matrix
    print(f"TF‑IDF matrix shape: {tfidf_matrix.shape}")

    # Save chunks, vectorizer, and matrix
    os.makedirs("rag_data", exist_ok=True)
    with open("rag_data/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open("rag_data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    # Save the matrix as a dense array if it's small, else keep sparse (but we'll need it dense for cosine later)
    # For small document sets, we can convert to dense.
    tfidf_dense = tfidf_matrix.toarray().astype('float32')
    np.save("rag_data/tfidf_embeddings.npy", tfidf_dense)
    print("✅ Saved TF‑IDF embeddings and vectorizer to rag_data/")

if __name__ == "__main__":
    main()