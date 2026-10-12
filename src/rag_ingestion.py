
from pathlib import Path
from typing import Dict, List

from src.llm_client import get_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

COLLECTION_NAME = "vehicle_pdm_knowledge"
STAGING_COLLECTION = "vehicle_pdm_knowledge_gemini_staging"

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed existing knowledge chunks with Gemini."""

    client = get_client()
    embeddings = []

    for index, text in enumerate(texts, start=1):
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=text,
            config={
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": EMBEDDING_DIMENSIONS,
            },
        )

        if not response.embeddings:
            raise RuntimeError(
                f"No embedding returned for chunk {index}."
            )

        values = response.embeddings[0].values

        if len(values) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Chunk {index}: expected {EMBEDDING_DIMENSIONS} "
                f"dimensions, received {len(values)}."
            )

        embeddings.append(list(values))

        print(
            f"[rag-ingestion] Embedded chunk "
            f"{index}/{len(texts)}",
            flush=True,
        )

    return embeddings


def migrate_existing_chunks() -> None:
    """Copy the existing chunks into a separate Gemini staging collection."""

    import chromadb

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore not found: {VECTORSTORE_DIR}"
        )

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    # Read the original collection without modifying it.
    original = client.get_collection(name=COLLECTION_NAME)

    original_data = original.get(
        include=["documents", "metadatas"]
    )

    ids = original_data["ids"]
    documents = original_data["documents"]
    metadatas = original_data["metadatas"]

    if not ids or not documents:
        raise RuntimeError("Original knowledge collection is empty.")

    if not (
        len(ids) == len(documents) == len(metadatas)
    ):
        raise RuntimeError(
            "Original collection returned inconsistent data lengths."
        )

    if any(not doc or not doc.strip() for doc in documents):
        raise RuntimeError(
            "Original collection contains an empty document."
        )

    print(
        f"[rag-ingestion] Original chunks: {len(documents)}",
        flush=True,
    )

    embeddings = embed_documents(documents)

    # Recreate only the staging collection.
    # The original collection is never deleted or modified.
    try:
        client.delete_collection(name=STAGING_COLLECTION)
    except Exception:
        pass

    staging = client.create_collection(
        name=STAGING_COLLECTION,
        metadata={
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
    )

    staging.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # Verify document and metadata preservation.
    if staging.count() != len(documents):
        raise RuntimeError(
            "Staging collection count does not match original."
        )

    check = staging.get(
        include=["documents", "metadatas"]
    )

    original_by_id = {
        item_id: (doc, meta)
        for item_id, doc, meta in zip(ids, documents, metadatas)
    }

    staged_by_id = {
        item_id: (doc, meta)
        for item_id, doc, meta in zip(
            check["ids"],
            check["documents"],
            check["metadatas"],
        )
    }

    if original_by_id != staged_by_id:
        raise RuntimeError(
            "Staging verification failed: documents or metadata differ."
        )

    print(
        f"[rag-ingestion] Migration verified | "
        f"chunks={staging.count()} | "
        f"dimensions={EMBEDDING_DIMENSIONS}",
        flush=True,
    )

    print(
        f"[rag-ingestion] Original collection preserved: "
        f"{COLLECTION_NAME}",
        flush=True,
    )


if __name__ == "__main__":
    migrate_existing_chunks()
