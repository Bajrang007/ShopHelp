"""
build_index.py
Chunks the policy markdown files, embeds them locally (no API calls,
no cost), and stores them in a persistent ChromaDB collection.

Run once with: python backend/build_index.py
Re-run any time the policy docs change.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "chroma_db")

POLICY_FILES = [
    "policy_returns.md",
    "policy_shipping.md",
    "policy_general_faq.md",
]


def chunk_markdown(text: str, source: str) -> list[dict]:
    """
    Splits a markdown file into chunks by ## headers. Each section
    becomes one chunk -- this keeps each chunk topically coherent
    (e.g. 'Refund timeline' is its own chunk, not mixed with 'Return window').
    """
    chunks = []
    sections = text.split("\n## ")
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # first section keeps the H1 title, rest get '## ' put back
        content = section if i == 0 else "## " + section
        if len(content) > 20:  # skip near-empty fragments
            chunks.append({"text": content, "source": source})
    return chunks


def main():
    print("Reading policy documents...")
    all_chunks = []
    for filename in POLICY_FILES:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_markdown(text, filename)
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Local embedding function -- runs on your machine, free, no API calls
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Wipe and recreate collection for a clean, repeatable build
    try:
        client.delete_collection("shopflow_policies")
    except Exception:
        pass

    collection = client.create_collection(
        name="shopflow_policies",
        embedding_function=embed_fn,
    )

    collection.add(
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    )

    print(f"\n✅ Indexed {len(all_chunks)} chunks into ChromaDB at {DB_DIR}")


if __name__ == "__main__":
    main()