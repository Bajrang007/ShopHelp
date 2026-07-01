"""
rag.py
Retrieval function over the ChromaDB policy index built in build_index.py.
This is what powers the search_policy tool the agent can call.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "chroma_db")

_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_client = chromadb.PersistentClient(path=DB_DIR)
_collection = _client.get_collection(
    name="shopflow_policies",
    embedding_function=_embed_fn,
)


def search_policy(query: str, n_results: int = 2) -> dict:
    """
    Searches the policy knowledge base for chunks relevant to the query.
    Returns the top matching chunks with their source file.
    """
    results = _collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    if not results["documents"] or not results["documents"][0]:
        return {"found": False, "message": "No relevant policy information found."}

    matches = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        matches.append({"source": meta["source"], "content": doc})

    return {"found": True, "matches": matches}


if __name__ == "__main__":
    # quick manual test
    test_queries = [
        "can I return swimwear",
        "how long does shipping take",
        "what payment methods do you accept",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = search_policy(q)
        print(result)