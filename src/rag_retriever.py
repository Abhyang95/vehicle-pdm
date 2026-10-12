
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from google.genai import types

from src.llm_client import get_client


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# Use the staging collection created by the updated ingestion code.
COLLECTION_NAME = "vehicle_pdm_knowledge_gemini_staging"

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


# ============================================================
# RAG Retriever
# ============================================================

class KnowledgeRetriever:
    """Retrieve knowledge using Gemini embeddings and ChromaDB."""

    def __init__(
        self,
        vectorstore_dir: Path = VECTORSTORE_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        init_start = perf_counter()
        self.vectorstore_dir = Path(vectorstore_dir)

        print(
            f"[rag] Retriever initialization started | "
            f"vectorstore={self.vectorstore_dir}",
            flush=True,
        )

        if not self.vectorstore_dir.exists():
            raise FileNotFoundError(
                f"Vectorstore directory not found: {self.vectorstore_dir}"
            )

        import chromadb

        self.client = chromadb.PersistentClient(
            path=str(self.vectorstore_dir)
        )

        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{collection_name}' not found. "
                "Run and verify the local Gemini ingestion first."
            ) from exc

        if self.collection.count() == 0:
            raise RuntimeError(
                f"Collection '{collection_name}' is empty."
            )

        print(
            f"[rag] ChromaDB ready | collection={collection_name} | "
            f"chunks={self.collection.count()}",
            flush=True,
        )

        # No SentenceTransformer model is loaded here.
        print(
            f"[rag] Using hosted embeddings | "
            f"model={EMBEDDING_MODEL_NAME} | "
            f"dimensions={EMBEDDING_DIMENSIONS} | "
            f"initialized in {perf_counter() - init_start:.2f}s",
            flush=True,
        )

    # ========================================================
    # Query embedding
    # ========================================================

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding compatible with stored document vectors."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        started = perf_counter()

        response = get_client().models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=query.strip(),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no query embedding.")

        values = response.embeddings[0].values

        if not values or len(values) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Unexpected Gemini query embedding dimensions: "
                f"{len(values) if values else 0}"
            )

        print(
            f"[rag] Query embedding completed in "
            f"{perf_counter() - started:.2f}s",
            flush=True,
        )

        return list(values)

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Dict]:
        """Retrieve relevant knowledge chunks from ChromaDB."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        started = perf_counter()
        query_embedding = self.embed_query(query)

        result_count = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []

        retrieved = []

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):
            metadata = metadata or {}

            retrieved.append({
                "text": document,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "distance": round(float(distance), 4),
            })

        print(
            f"[rag] Retrieval completed in "
            f"{perf_counter() - started:.2f}s | "
            f"results={len(retrieved)}",
            flush=True,
        )

        return retrieved


# ============================================================
# Cached retriever
# ============================================================

@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetriever:
    """Reuse one lightweight retriever per process."""

    print("[rag] Creating cached retriever", flush=True)
    return KnowledgeRetriever()


# ============================================================
# Convenience function
# ============================================================

def retrieve_knowledge(
    query: str,
    top_k: int = 4,
) -> List[Dict]:
    """Retrieve knowledge using the cached retriever."""

    return get_retriever().retrieve(
        query=query,
        top_k=top_k,
    )


# ============================================================
# CLI testing
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "What does critical subsystem health mean?",
        "Why might predicted remaining useful life change?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        for index, result in enumerate(
            retrieve_knowledge(query, top_k=4), start=1
        ):
            print(f"\nResult {index}: {result['source']}")
            print(f"Distance: {result['distance']}")
            print(result["text"][:500])
