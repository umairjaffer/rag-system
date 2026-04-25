"""
app/models.py
-------------
Pydantic models (schemas) for all API request and response bodies.

Keeping schemas in one place makes it easy to:
  - Understand what data flows in and out of every endpoint.
  - Update validation rules without touching router logic.
  - Generate accurate OpenAPI docs automatically via FastAPI.
"""

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload endpoint schemas
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """
    Returned after a PDF is successfully uploaded and indexed.
    """

    message: str = Field(..., description="Human-readable status message.")
    filename: str = Field(..., description="Original name of the uploaded file.")
    chunks_indexed: int = Field(
        ..., description="Number of text chunks added to the FAISS vector store."
    )


# ---------------------------------------------------------------------------
# Query endpoint schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """
    Payload sent by the client when asking a question.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to answer from the uploaded document(s).",
        examples=["What is the main topic of the document?"],
    )


class SourceChunk(BaseModel):
    """
    Metadata about a single document chunk that was used to answer the question.
    Helps users verify which part of the PDF the answer came from.
    """

    source_file: str = Field(
        ..., description="Original PDF filename the chunk was extracted from."
    )
    page: int = Field(..., description="Page number (1-indexed) within the PDF.")
    chunk_text: str = Field(
        ..., description="The actual text content of the retrieved chunk."
    )


class QueryResponse(BaseModel):
    """
    Returned after processing a user question against the vector store.
    """

    question: str = Field(..., description="The original question that was asked.")
    answer: str = Field(..., description="LLM-generated answer based on retrieved chunks.")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="List of document chunks that were retrieved and used as context.",
    )


# ---------------------------------------------------------------------------
# Health check schema
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """
    Returned by the health check endpoint.
    Used by load balancers and monitoring tools to verify service readiness.
    """

    status: str = Field(..., description="'ok' when the service is healthy.")
    version: str = Field(..., description="API version string.")
    vector_store_loaded: bool = Field(
        ..., description="True if a FAISS index is loaded and ready for queries."
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information.",
    )