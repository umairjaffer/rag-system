"""
app/rag_engine.py
-----------------
Core RAG (Retrieval-Augmented Generation) engine.

This module encapsulates all LangChain logic:
  - PDF loading and text splitting
  - Embedding with a local HuggingFace sentence-transformers model
  - FAISS vector store management (build, persist, load)
  - Retrieval + LLM chain using Groq for fast inference

Design decisions:
  - The embedding model is loaded once at startup (expensive GPU/CPU op).
  - The FAISS index is persisted to disk so it survives server restarts.
  - Each uploaded PDF is merged into the existing index so users can
    ask questions across multiple documents.
  - Source metadata (filename, page number) is stored alongside every
    chunk so the API can tell the user exactly where the answer came from.
"""

import logging
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.models import QueryResponse, SourceChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided context.

If the answer cannot be found in the context, say "I don't have enough information in the provided documents to answer this question." Do not make up answers.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: list[Document]) -> str:
    """
    Concatenates document chunks into a single context string for the LLM prompt.

    Args:
        docs: List of retrieved LangChain Document objects.

    Returns:
        A single string with all chunk texts joined by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class RAGEngine:
    """
    Encapsulates the full Retrieval-Augmented Generation pipeline.

    Responsibilities:
        1. Load and chunk PDF documents.
        2. Embed chunks with a local HuggingFace model.
        3. Store / update a FAISS vector index on disk.
        4. Retrieve relevant chunks and generate answers via Groq LLM.
    """

    def __init__(self) -> None:
        """
        Initialises the embedding model, LLM, and loads any existing
        FAISS index from disk. The embedding model loading can take
        several seconds on first run while the model weights download.
        """
        logger.info("Initialising RAG engine ...")

        # Ensure storage directories exist
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)

        # --- Embedding model (runs locally, no API key required) ---
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # --- Groq LLM ---
        logger.info("Initialising Groq LLM: %s", settings.groq_model_name)
        self.llm = ChatGroq(
            model=settings.groq_model_name,
            api_key=settings.groq_api_key,
            temperature=0.1,
            max_retries=2,
        )

        # --- Text splitter ---
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # --- FAISS vector store ---
        self.vector_store: FAISS | None = None
        self._load_existing_index()

        # --- RAG prompt ---
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        logger.info("RAG engine initialised successfully.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing_index(self) -> None:
        """
        Loads a previously saved FAISS index from disk if one exists.
        This allows the server to resume without re-indexing all PDFs
        after a restart.
        """
        index_file = Path(settings.faiss_index_path) / "index.faiss"
        if index_file.exists():
            logger.info("Found existing FAISS index at '%s'. Loading ...", settings.faiss_index_path)
            try:
                self.vector_store = FAISS.load_local(
                    settings.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded successfully.")
            except Exception as exc:
                logger.warning("Could not load existing FAISS index: %s. Starting fresh.", exc)
                self.vector_store = None
        else:
            logger.info("No existing FAISS index found. A new one will be created on first upload.")

    def _save_index(self) -> None:
        """
        Persists the current FAISS vector store to disk so it survives
        server restarts without needing to re-embed all documents.
        """
        if self.vector_store is not None:
            self.vector_store.save_local(settings.faiss_index_path)
            logger.info("FAISS index saved to '%s'.", settings.faiss_index_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_pdf(self, file_path: str, filename: str) -> int:
        """
        Loads a PDF, splits it into chunks, embeds them, and merges them
        into the FAISS vector store (creating it if it does not exist yet).

        Args:
            file_path: Absolute path to the saved PDF file on disk.
            filename:  Original filename of the PDF (stored in metadata
                       so users can see which document each chunk came from).

        Returns:
            Number of chunks that were added to the vector store.

        Raises:
            ValueError: If the PDF contains no extractable text.
            Exception:  Propagates unexpected errors from PyPDFLoader or FAISS.
        """
        logger.info("Indexing PDF: %s", filename)

        # Load the PDF (each page becomes one Document with page metadata)
        loader = PyPDFLoader(file_path)
        pages: list[Document] = loader.load()

        if not pages:
            raise ValueError(f"No content could be extracted from '{filename}'.")

        # Attach the original filename to every page's metadata so it
        # propagates to every chunk after splitting.
        for page in pages:
            page.metadata["source"] = filename
            # PyPDFLoader sets page as 0-indexed; convert to 1-indexed for display.
            page.metadata["page"] = page.metadata.get("page", 0) + 1

        # Split pages into smaller overlapping chunks
        chunks: list[Document] = self.text_splitter.split_documents(pages)

        if not chunks:
            raise ValueError(f"Text splitting produced no chunks for '{filename}'.")

        logger.info("Split '%s' into %d chunks.", filename, len(chunks))

        # Build or merge into the FAISS vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("Created new FAISS index with %d chunks.", len(chunks))
        else:
            self.vector_store.add_documents(chunks)
            logger.info("Added %d chunks to existing FAISS index.", len(chunks))

        # Persist to disk immediately so data is not lost on crash
        self._save_index()

        return len(chunks)

    def query(self, question: str) -> QueryResponse:
        """
        Answers a question using the RAG pipeline:
          1. Embed the question and retrieve the top-k most similar chunks.
          2. Pass the chunks as context to the Groq LLM.
          3. Return the answer along with source metadata for verification.

        Args:
            question: The user's natural language question.

        Returns:
            QueryResponse containing the answer and source chunks.

        Raises:
            ValueError: If no vector store has been built yet (no PDFs uploaded).
            Exception:  Propagates unexpected LLM or retrieval errors.
        """
        if self.vector_store is None:
            raise ValueError(
                "No documents have been indexed yet. Please upload a PDF first."
            )

        # Set up retriever with MMR search for diverse results
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.retriever_k, "fetch_k": settings.retriever_k * 2},
        )

        # Retrieve the most relevant chunks (also used to build source list)
        retrieved_docs: list[Document] = retriever.invoke(question)

        # Build the LCEL chain: retrieve -> format -> prompt -> LLM -> parse
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("Running RAG chain for question: '%s'", question[:80])
        answer: str = chain.invoke(question)

        # Build source metadata list for the response
        sources: list[SourceChunk] = [
            SourceChunk(
                source_file=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 0),
                chunk_text=doc.page_content,
            )
            for doc in retrieved_docs
        ]

        return QueryResponse(question=question, answer=answer, sources=sources)

    @property
    def is_ready(self) -> bool:
        """Returns True if the vector store has been loaded and is ready to query."""
        return self.vector_store is not None