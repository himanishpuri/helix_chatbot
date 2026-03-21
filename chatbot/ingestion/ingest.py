# ingestion/ingest.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────
MARKDOWN_FILE = os.path.join(os.path.dirname(__file__), "college_data.md")
CHROMA_PATH   = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
COLLECTION    = "college_kb"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
# ─────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]

def main():
    with open(MARKDOWN_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    print(f"[1/4] Loaded markdown — {len(raw)} characters")

    chunks = chunk_text(raw)
    print(f"[2/4] Created {len(chunks)} chunks")

    print("[3/4] Loading embedding model (downloads once ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"[4/4] Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    print("✓ Ingestion complete")

if __name__ == "__main__":
    main()