"""
app/dependencies.py
-------------------
FastAPI dependency functions used across multiple routers.

Dependencies are resolved by FastAPI's dependency injection system,
which means they are created once per request (or once per process
if they hold expensive resources like models).

This pattern keeps routers clean and makes it easy to swap
implementations (e.g., swap FAISS for another vector store) without
changing router code.
"""

import logging

from fastapi import HTTPException, status

from app.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Module-level singleton so the embedding model and LLM are loaded only once.
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """
    Returns the module-level RAGEngine singleton.

    The engine is created when the FastAPI application starts up
    (see lifespan in main.py). Raises an internal server error if
    it has not been initialised yet, which should never happen in
    normal operation.
    """
    if _rag_engine is None:
        logger.error("RAGEngine has not been initialised.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine is not ready. Please try again in a moment.",
        )
    return _rag_engine


def set_rag_engine(engine: RAGEngine) -> None:
    """
    Stores the RAGEngine singleton.

    Called once during application startup so all subsequent requests
    share the same loaded embedding model and LLM client.
    """
    global _rag_engine
    _rag_engine = engine