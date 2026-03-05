"""
embedder.py
-----------
Converts text chunks into numerical vectors (embeddings).
We use sentence-transformers which runs 100% locally and is free.

Model: all-MiniLM-L6-v2
- Small and fast (only 80MB)
- Great quality for semantic similarity
- Outputs 384-dimensional vectors
"""

from sentence_transformers import SentenceTransformer

# Load model once when module is imported (avoid reloading on every call)
print("[Embedder] Loading embedding model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Embedder] Model loaded!")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.
    Returns a list of float arrays.
    """
    embeddings = _model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Used when a user asks a question — we embed it to search the DB.
    """
    embedding = _model.encode([query], show_progress_bar=False)
    return embedding[0].tolist()
