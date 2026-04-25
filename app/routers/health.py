"""
app/routers/health.py
---------------------
Health check endpoint for monitoring and load balancer probes.

Production services need a reliable way to verify that:
  - The application process is running.
  - The RAG engine (embedding model + LLM client) is initialised.
  - The FAISS vector store is loaded and ready for queries.

Load balancers (AWS ALB, GCP GLB, nginx, etc.) and orchestrators
(Kubernetes liveness/readiness probes) hit this endpoint periodically.
A non-200 response removes the instance from the pool until it recovers.
"""

import logging

from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_rag_engine
from app.models import HealthResponse
from app.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the current health status of the API. "
        "Use this endpoint for load balancer probes and uptime monitoring."
    ),
)
async def health_check(
    engine: RAGEngine = Depends(get_rag_engine),
) -> HealthResponse:
    """
    Returns service health status.

    Args:
        engine: RAGEngine injected by FastAPI (confirms engine is alive).

    Returns:
        HealthResponse with status, version, and vector store readiness.
    """
    logger.debug("Health check requested.")

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        vector_store_loaded=engine.is_ready,
        details={
            "llm_model": settings.groq_model_name,
            "embedding_model": settings.embedding_model_name,
            "index_path": settings.faiss_index_path,
        },
    )