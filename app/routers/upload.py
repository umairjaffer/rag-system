"""
app/routers/upload.py
---------------------
Router for the PDF upload endpoint.

Flow:
  1. Client sends a multipart/form-data POST request with a PDF file.
  2. The file is validated (must be a PDF, must not be empty).
  3. The file is saved to the UPLOAD_DIR on disk.
  4. The RAG engine indexes the file (chunks + embeds + stores in FAISS).
  5. A response is returned with the number of indexed chunks.

Error handling:
  - 400 Bad Request if the file is not a PDF or is empty.
  - 500 Internal Server Error if indexing fails unexpectedly.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import settings
from app.dependencies import get_rag_engine
from app.models import UploadResponse
from app.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF and index it into the vector store",
    description=(
        "Upload a PDF file. The file is split into chunks, embedded using a local "
        "HuggingFace model, and stored in the FAISS vector store. "
        "After uploading, the document is available for question answering."
    ),
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload and index."),
    engine: RAGEngine = Depends(get_rag_engine),
) -> UploadResponse:
    """
    Upload and index a PDF document.

    Args:
        file:   The uploaded file from the multipart form.
        engine: RAGEngine injected by FastAPI's dependency system.

    Returns:
        UploadResponse with filename and number of indexed chunks.
    """

    # --- Validate file type ---
    if file.content_type not in ("application/pdf",) and not (
        file.filename and file.filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename.",
        )

    # --- Read file contents ---
    contents: bytes = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # --- Save to disk ---
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    save_path = upload_path / file.filename

    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info("Saved uploaded file to '%s'.", save_path)
    except OSError as exc:
        logger.exception("Failed to save uploaded file '%s'.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save the uploaded file: {exc}",
        ) from exc

    # --- Index the PDF ---
    try:
        chunks_indexed = engine.index_pdf(
            file_path=str(save_path),
            filename=file.filename,
        )
    except ValueError as exc:
        # Clean up the saved file if indexing fails due to empty/invalid content
        if save_path.exists():
            os.remove(save_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while indexing '%s'.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while indexing the PDF.",
        ) from exc

    return UploadResponse(
        message="PDF uploaded and indexed successfully.",
        filename=file.filename,
        chunks_indexed=chunks_indexed,
    )