import os
import pickle

KNOWLEDGE_DIR = "my_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Try to break at space, newline, or punctuation
            for i in range(end, max(start, end-150), -1):
                if i < text_len and text[i] in (' ', '\n', '.', '!', '?'):
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
    os.makedirs("rag_data", exist_ok=True)
    with open("rag_data/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("✅ Saved chunks to rag_data/chunks.pkl")

if __name__ == "__main__":
    main()