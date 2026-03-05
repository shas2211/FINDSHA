"""
vectorstore.py
--------------
Handles all ChromaDB operations:
- Storing embedded chunks
- Searching for similar chunks given a query

ChromaDB stores vectors on disk (./chroma_db folder).
Each document gets its own "collection" so you can query per-document.
"""

import chromadb
import hashlib
import os

# Persistent client — data survives restarts
_client = chromadb.PersistentClient(path="./chroma_db")


def _get_collection(collection_name: str):
    """Get or create a ChromaDB collection for a document."""
    # Sanitize collection name (chroma only allows alphanumeric + underscore)
    safe_name = "".join(c if c.isalnum() else "_" for c in collection_name)
    return _client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )


def store_chunks(chunks: list[str], embeddings: list[list[float]], doc_name: str):
    """
    Store text chunks and their embeddings in ChromaDB.
    Each chunk gets a unique ID based on its content hash.
    """
    collection = _get_collection(doc_name)

    # Generate unique IDs for each chunk
    ids = [hashlib.md5(f"{doc_name}_{i}_{chunk[:50]}".encode()).hexdigest()
           for i, chunk in enumerate(chunks)]

    # Add to ChromaDB
    # ChromaDB stores: id, embedding, document text, and optional metadata
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]
    )
    print(f"[VectorStore] Stored {len(chunks)} chunks for '{doc_name}'")


def search_similar(query_embedding: list[float], doc_name: str, top_k: int = 5) -> list[str]:
    """
    Find the top_k most similar chunks to the query embedding.
    Returns the actual text of those chunks.
    """
    collection = _get_collection(doc_name)

    # Check if collection has any data
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents"]
    )

    # results["documents"] is a list of lists (one per query)
    chunks = results["documents"][0] if results["documents"] else []
    print(f"[VectorStore] Retrieved {len(chunks)} relevant chunks")
    return chunks


def list_documents() -> list[str]:
    """Return all document collections stored in ChromaDB."""
    collections = _client.list_collections()
    return [col.name for col in collections]


def delete_document(doc_name: str):
    """Delete a document's collection from ChromaDB."""
    safe_name = "".join(c if c.isalnum() else "_" for c in doc_name)
    try:
        _client.delete_collection(safe_name)
        print(f"[VectorStore] Deleted collection: {safe_name}")
    except Exception as e:
        print(f"[VectorStore] Could not delete: {e}")
