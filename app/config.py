"""
app/config.py
-------------
Centralized, type-safe application configuration using Pydantic Settings.

All values are read from environment variables or the .env file at startup.
Developers can override any setting by setting the corresponding env var.

Usage:
    from app.config import settings
    print(settings.groq_model_name)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    Pydantic validates types automatically on startup so misconfiguration
    is caught early rather than at runtime.
    """

    # --- Groq LLM ---
    groq_api_key: str
    groq_model_name: str = "llama-3.3-70b-versatile"

    # --- Embedding Model (HuggingFace, runs locally) ---
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # --- Storage paths ---
    faiss_index_path: str = "faiss_index"
    upload_dir: str = "uploads"

    # --- Text splitting parameters ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval parameters ---
    retriever_k: int = 4

    # --- FastAPI metadata ---
    app_title: str = "RAG PDF Question Answering API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Upload PDFs and ask questions using Groq LLM + FAISS vector store"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    Using lru_cache ensures the .env file is read only once for the
    lifetime of the process, which is important for performance in
    production where settings do not change at runtime.
    """
    return Settings()


# Module-level singleton for convenience imports
settings = get_settings()