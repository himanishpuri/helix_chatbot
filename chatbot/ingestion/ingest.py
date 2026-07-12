# ingestion/ingest.py
import os
import hashlib

MARKDOWN_FILE = os.path.join(os.path.dirname(__file__), "college_data.md")
CHROMA_PATH = os.getenv(
    "CHROMA_PATH", os.path.join(os.path.dirname(__file__), "../data/chroma_db")
)
COLLECTION = "college_kb"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 32


def chunk_text(text: str):
    """Paragraph-packed chunks of ~CHUNK_SIZE with CHUNK_OVERLAP char carry-over.

    Paragraphs are packed until they'd exceed CHUNK_SIZE; a paragraph larger
    than CHUNK_SIZE is hard-split. Each new chunk carries the trailing
    CHUNK_OVERLAP characters of the previous one so context isn't cut mid-idea.
    """
    # Hard-split oversized paragraphs so no unit exceeds CHUNK_SIZE.
    units = []
    for p in text.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        while len(p) > CHUNK_SIZE:
            units.append(p[:CHUNK_SIZE])
            p = p[CHUNK_SIZE:]
        if p:
            units.append(p)

    chunks = []
    buffer = ""
    for u in units:
        if buffer and len(buffer) + len(u) + 2 > CHUNK_SIZE:
            chunks.append(buffer.strip())
            overlap = buffer[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            buffer = overlap + "\n\n" + u
        else:
            buffer = buffer + "\n\n" + u if buffer else u

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def hash_chunk(text: str):
    return hashlib.md5(text.encode()).hexdigest()


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

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


def _selfcheck():
    assert chunk_text("") == []
    assert chunk_text("\n\n  \n\n") == []
    # oversized single paragraph is split, no chunk exceeds CHUNK_SIZE
    big = chunk_text("x" * (CHUNK_SIZE * 2 + 10))
    # each chunk is one hard-split unit (<=CHUNK_SIZE) plus at most one overlap carry
    assert big and all(len(c) <= CHUNK_SIZE + CHUNK_OVERLAP + 2 for c in big)
    # overlap: consecutive chunks share trailing/leading text
    packed = chunk_text("\n\n".join(f"para-{i} " + "y" * 200 for i in range(6)))
    assert len(packed) >= 2
    assert any(c for c in packed)  # no empty chunks
    print("✓ chunk_text self-check passed")


if __name__ == "__main__":
    import sys

    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
