# ingestion/ingest.py
import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

MARKDOWN_FILE = os.path.join(os.path.dirname(__file__), "college_data.md")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
COLLECTION = "college_kb"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 32


def chunk_text(text: str):
    paragraphs = text.split("\n\n")
    chunks = []
    buffer = ""

    for p in paragraphs:
        if len(buffer) + len(p) < CHUNK_SIZE:
            buffer += p + "\n\n"
        else:
            chunks.append(buffer.strip())
            buffer = p + "\n\n"

    if buffer:
        chunks.append(buffer.strip())

    return chunks


def hash_chunk(text: str):
    return hashlib.md5(text.encode()).hexdigest()


def main():
    with open(MARKDOWN_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = chunk_text(raw)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        embeddings = model.encode(batch).tolist()

        ids = [hash_chunk(c) for c in batch]

        metadata = [
            {"source": "college_data.md", "chunk_index": i + j}
            for j in range(len(batch))
        ]

        collection.add(
            documents=batch, embeddings=embeddings, metadatas=metadata, ids=ids
        )

    print("✓ Ingestion complete")


if __name__ == "__main__":
    main()
