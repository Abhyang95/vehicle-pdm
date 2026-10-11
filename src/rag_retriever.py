
from pathlib import Path
from typing import List, Dict




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

    The retriever is intentionally independent from any LLM.
    This allows retrieval quality to be tested separately.
    """

    def __init__(
        self,
        vectorstore_dir: Path = VECTORSTORE_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.vectorstore_dir = Path(
            vectorstore_dir
        )

        if not self.vectorstore_dir.exists():
            raise FileNotFoundError(
                "Vectorstore directory not found: "
                f"{self.vectorstore_dir}"
            )
        import chromadb
        from sentence_transformers import SentenceTransformer
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

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    # ========================================================
    # Query embedding
    # ========================================================

    def embed_query(
        self,
        query: str,
    ):
        """
        Convert a user query into a normalized embedding.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        embedding = self.embedding_model.encode(
            query.strip(),
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Dict]:
        """
        Retrieve the most relevant knowledge chunks.

        Returns dictionaries containing:
            - text
            - source
            - chunk_index
            - distance
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        query_embedding = self.embed_query(
            query
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

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                {
                    "text": document,
                    "source": metadata.get(
                        "source",
                        "unknown",
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index",
                        -1,
                    ),
                    "distance": round(
                        float(distance),
                        4,
                    ),
                }
            )

        return retrieved


# ============================================================
# Convenience function
# ============================================================

def retrieve_knowledge(
    query: str,
    top_k: int = 4,
) -> List[Dict]:
    """
    Convenience wrapper around KnowledgeRetriever.
    """

    retriever = KnowledgeRetriever()

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
):
    """
    Print retrieval results in a readable format.
    """

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)

    print(f"\nQuery:\n{query}")

    print(
        f"\nRetrieved chunks: {len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 70)

        print(
            f"Result {index}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Chunk: {result['chunk_index']}"
        )

        print(
            f"Distance: {result['distance']}"
        )

        print("\nContent:")

        print(
            result["text"]
        )


if __name__ == "__main__":

    test_query = (
        "What does critical subsystem health mean?"
    )

    results = retrieve_knowledge(
        query=test_query,
        top_k=4,
    )

    print_results(
        query=test_query,
        results=results,
    )
