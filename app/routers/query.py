"""
app/routers/query.py
--------------------
Router for the question-answering endpoint.

Flow:
  1. Client sends a JSON POST request with a question string.
  2. The RAG engine retrieves relevant chunks from the FAISS vector store.
  3. The Groq LLM generates an answer grounded in those chunks.
  4. The response includes the answer and source metadata (filename, page,
     chunk text) so the user can verify where the answer came from.

Error handling:
  - 400 Bad Request if no documents have been indexed yet.
  - 500 Internal Server Error for unexpected LLM or retrieval failures.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_rag_engine
from app.models import QueryRequest, QueryResponse
from app.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about uploaded documents",
    description=(
        "Submit a natural language question. The system retrieves the most "
        "relevant chunks from the FAISS vector store and uses Groq LLM to "
        "generate an answer. The response includes source metadata so you "
        "can verify which document and page the answer came from."
    ),
)
async def ask_question(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine),
) -> QueryResponse:
    """
    Answer a question using the RAG pipeline.

    Args:
        request: Validated request body containing the question string.
        engine:  RAGEngine injected by FastAPI's dependency system.

    Returns:
        QueryResponse with the answer and list of source chunks.
    """

    try:
        response = engine.query(question=request.question)
    except ValueError as exc:
        # Raised when no documents have been indexed yet
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing query: '%s'", request.question[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your question.",
        ) from exc

    return response