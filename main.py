"""
main.py
-------
FastAPI application entry point.

This file is intentionally kept minimal. It:
  1. Creates the FastAPI app with metadata and CORS settings.
  2. Manages the application lifespan (startup / shutdown) to initialise
     and clean up the RAGEngine exactly once per process.
  3. Registers all routers.
  4. Provides a root redirect to the auto-generated API docs.

Run the development server with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

For production, use multiple workers:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.dependencies import set_rag_engine
from app.rag_engine import RAGEngine
from app.routers import health, query, upload

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Configure once at the entry point so every module that calls
# logging.getLogger(__name__) inherits this configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (replaces deprecated @app.on_event decorators)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages startup and shutdown of shared resources.

    On startup:
        - Initialises the RAGEngine (loads embedding model, Groq LLM client,
          and any previously saved FAISS index from disk).
        - Stores it in the dependency module so all requests share one instance.

    On shutdown:
        - Logs a clean shutdown message (add cleanup logic here if needed,
          e.g. flushing caches or closing database connections).
    """
    logger.info("Starting up RAG API server ...")
    engine = RAGEngine()
    set_rag_engine(engine)
    logger.info("RAG API server is ready.")

    yield  # Application runs here; everything below runs on shutdown

    logger.info("Shutting down RAG API server ...")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
# Allows the Next.js frontend (typically on localhost:3000 in development
# or a separate domain in production) to call this API.
# In production, replace allow_origins with your actual domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(query.router)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirects the root URL to the interactive Swagger UI docs."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# This block only runs when you execute the file directly:
#     python main.py
#
# It does NOT run when uvicorn imports the module itself (e.g. in production
# with multiple workers), which is the correct behaviour.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",       # module:app_variable (string form enables --reload)
        host="0.0.0.0",
        port=8000,
        reload=True,      # auto-restart on file changes (dev only)
        log_level="info",
    )