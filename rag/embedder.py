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

# Lazy load — model loads on first use, not at import (fixes Render port binding timeout)
_model = None

def _get_model():
    global _model
    if _model is None:
        print("[Embedder] Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Embedder] Model loaded!")
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.
    Returns a list of float arrays.
    """
    embeddings = _get_model().encode(texts, show_progress_bar=False)
    return embeddings.tolist()

def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Used when a user asks a question — we embed it to search the DB.
    """
    embedding = _get_model().encode([query], show_progress_bar=False)
    return embedding[0].tolist()
