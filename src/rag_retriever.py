
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Dict, List

# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

COLLECTION_NAME = "vehicle_pdm_knowledge"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# RAG Retriever
# ============================================================

class KnowledgeRetriever:
    """
    Retrieve relevant project knowledge from ChromaDB.

    The retriever is independent from any LLM.
    Its instance is cached so the embedding model and
    ChromaDB client are not recreated for every request.
    """

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
                f"Vectorstore directory not found: "
                f"{self.vectorstore_dir}"
            )

        # ----------------------------------------------------
        # Initialize ChromaDB
        # ----------------------------------------------------

        chroma_start = perf_counter()
        print("[rag] ChromaDB initialization started", flush=True)

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
                f"ChromaDB collection '{collection_name}' "
                "was not found. Run rag_ingestion.py first."
            ) from exc

        print(
            f"[rag] ChromaDB initialized in "
            f"{perf_counter() - chroma_start:.2f}s | "
            f"collection={collection_name} | "
            f"chunks={self.collection.count()}",
            flush=True,
        )

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        model_start = perf_counter()
        print(
            f"[rag] Embedding model loading started | "
            f"model={EMBEDDING_MODEL_NAME}",
            flush=True,
        )

        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print(
            f"[rag] Embedding model loaded in "
            f"{perf_counter() - model_start:.2f}s",
            flush=True,
        )

        print(
            f"[rag] Retriever initialization completed in "
            f"{perf_counter() - init_start:.2f}s",
            flush=True,
        )

    # ========================================================
    # Query embedding
    # ========================================================

    def embed_query(self, query: str) -> List[float]:
        """Convert a user query into a normalized embedding."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        embedding_start = perf_counter()

        print("[rag] Query embedding started", flush=True)

        embedding = self.embedding_model.encode(
            query.strip(),
            normalize_embeddings=True,
        )

        result = embedding.tolist()

        print(
            f"[rag] Query embedding completed in "
            f"{perf_counter() - embedding_start:.2f}s",
            flush=True,
        )

        return result

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Dict]:
        """
        Retrieve relevant knowledge chunks.

        Each result contains:
            - text
            - source
            - chunk_index
            - distance
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        retrieval_start = perf_counter()

        # ----------------------------------------------------
        # Embed query
        # ----------------------------------------------------

        query_embedding = self.embed_query(query)

        # ----------------------------------------------------
        # Query ChromaDB
        # ----------------------------------------------------

        chroma_query_start = perf_counter()

        print(
            f"[rag] ChromaDB query started | top_k={top_k}",
            flush=True,
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        print(
            f"[rag] ChromaDB query completed in "
            f"{perf_counter() - chroma_query_start:.2f}s",
            flush=True,
        )

        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            retrieved.append(
                {
                    "text": document,
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index", -1),
                    "distance": round(float(distance), 4),
                }
            )

        print(
            f"[rag] Retrieval completed in "
            f"{perf_counter() - retrieval_start:.2f}s | "
            f"results={len(retrieved)}",
            flush=True,
        )

        return retrieved


# ============================================================
# Cached retriever
# ============================================================

@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetriever:
    """
    Initialize one retriever per process and reuse it.

    If initialization fails, lru_cache does not cache the
    failed call, so a later request can retry.
    """

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

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer.")

    retriever = get_retriever()

    return retriever.retrieve(
        query=query,
        top_k=top_k,
    )


# ============================================================
# CLI testing
# ============================================================

def print_results(
    query: str,
    results: List[Dict],
) -> None:
    """Print retrieval results in a readable format."""

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)

    print(f"\nQuery:\n{query}")
    print(f"\nRetrieved chunks: {len(results)}")

    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 70)
        print(f"Result {index}")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Distance: {result['distance']}")
        print("\nContent:")
        print(result["text"])


if __name__ == "__main__":
    test_query = "What does critical subsystem health mean?"

    # Run twice to verify that the cached retriever is reused.
    for run_number in range(1, 3):
        print(f"\n--- Retrieval run {run_number} ---")

        results = retrieve_knowledge(
            query=test_query,
            top_k=4,
        )

        print_results(
            query=test_query,
            results=results,
        )
