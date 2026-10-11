
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_DIR = PROJECT_ROOT / "docs" / "knowledge"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

COLLECTION_NAME = "vehicle_pdm_knowledge"

# Small, lightweight embedding model suitable for local CPU inference.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking configuration.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ============================================================
# Knowledge document loading
# ============================================================

def load_markdown_documents() -> List[Dict]:
    """
    Load all Markdown knowledge documents.

    Returns:
        List of dictionaries containing:
        - source
        - text
    """

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            f"No Markdown knowledge documents found in {KNOWLEDGE_DIR}"
        )

    return documents


# ============================================================
# Markdown-aware chunking
# ============================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    The splitter first attempts to preserve paragraph boundaries,
    then falls back to character-based splitting for paragraphs
    that exceed the configured chunk size.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:

        # Normal case: paragraph fits into current chunk.
        if len(current) + len(paragraph) + 2 <= chunk_size:
            if current:
                current += "\n\n" + paragraph
            else:
                current = paragraph
            continue

        # Save current chunk before processing the large paragraph.
        if current:
            chunks.append(current.strip())

        # If the paragraph itself is small enough, start a new chunk.
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        # Large paragraph fallback.
        start = 0

        while start < len(paragraph):
            end = min(
                start + chunk_size,
                len(paragraph),
            )

            piece = paragraph[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= len(paragraph):
                break

            start = max(
                end - chunk_overlap,
                start + 1,
            )

        current = ""

    if current:
        chunks.append(current.strip())

    return [
        chunk
        for chunk in chunks
        if chunk
    ]


def build_chunks(
    documents: List[Dict],
) -> List[Dict]:
    """
    Convert source documents into embedding-ready chunks.

    Each chunk contains:
        - id
        - text
        - source
        - chunk_index
    """

    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = split_text(text)

        for index, chunk in enumerate(document_chunks):
            chunks.append(
                {
                    "id": f"{source}:{index}",
                    "text": chunk,
                    "source": source,
                    "chunk_index": index,
                }
            )

    return chunks


# ============================================================
# ChromaDB
# ============================================================

def get_chroma_client():
    """
    Create a persistent ChromaDB client.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )


def create_vectorstore(
    chunks: List[Dict],
):
    """
    Create or replace the project knowledge collection.
    """

    client = get_chroma_client()

    # Remove the previous collection if it exists.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Vehicle Predictive Maintenance "
                "knowledge base"
            )
        },
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    collection.add(
        ids=[
            chunk["id"]
            for chunk in chunks
        ],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    return collection


# ============================================================
# Pipeline
# ============================================================

def build_knowledge_base():
    """
    Complete knowledge-base ingestion pipeline.
    """

    print("=" * 60)
    print("VEHICLE PREDICTIVE MAINTENANCE RAG INGESTION")
    print("=" * 60)

    print(f"\nKnowledge directory:")
    print(f"  {KNOWLEDGE_DIR}")

    print(f"\nVectorstore directory:")
    print(f"  {VECTORSTORE_DIR}")

    print(f"\nEmbedding model:")
    print(f"  {EMBEDDING_MODEL_NAME}")

    print("\nLoading documents...")

    documents = load_markdown_documents()

    print(
        f"Loaded {len(documents)} knowledge documents."
    )

    for document in documents:
        print(
            f"  - {document['source']}"
        )

    print("\nBuilding chunks...")

    chunks = build_chunks(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\nCreating embeddings and vectorstore...")

    collection = create_vectorstore(
        chunks
    )

    print(
        f"Stored {collection.count()} chunks "
        f"in ChromaDB."
    )

    print("\nRAG ingestion completed successfully.")

    return collection


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    build_knowledge_base()
