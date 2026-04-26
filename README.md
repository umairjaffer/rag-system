# RAG PDF Question Answering API

A production-ready **Retrieval-Augmented Generation (RAG)** API built with:

- **FastAPI** — async web framework with auto-generated Swagger docs
- **LangChain 0.3** — LLM orchestration using modern LCEL chains
- **Groq** — ultra-fast LLM inference (`llama-3.3-70b-versatile`)
- **HuggingFace sentence-transformers** — local embedding model (no API cost)
- **FAISS** — in-process vector store persisted to disk

---

## Project Structure

```
rag-system/
├── main.py                  # FastAPI app entry point (outside app/)
├── .env                     # Environment variables (never commit this)
├── .env.example             # Template for .env
├── .gitignore
├── requirements.txt
└── app/
    ├── __init__.py
    ├── config.py            # Typed settings via pydantic-settings
    ├── dependencies.py      # FastAPI dependency injection helpers
    ├── models.py            # Pydantic request / response schemas
    ├── rag_engine.py        # Core RAG logic (PDF load, embed, retrieve, generate)
    └── routers/
        ├── __init__.py
        ├── health.py        # GET  /health/
        ├── upload.py        # POST /upload/
        └── query.py         # POST /query/
```

---

## Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/umairjaffer/rag-system.git
cd rag-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free Groq API key at: https://console.groq.com

### 5. Start the development server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 — it redirects to the Swagger UI at http://localhost:8000/docs

---

## API Endpoints

| Method | Path       | Description                            |
|--------|------------|----------------------------------------|
| GET    | /health/   | Health check (vector store status)     |
| POST   | /upload/   | Upload a PDF and index it into FAISS   |
| POST   | /query/    | Ask a question about uploaded PDFs     |
| GET    | /docs      | Swagger UI (interactive API explorer)  |
| GET    | /redoc     | ReDoc UI                               |

### Upload a PDF

```bash
curl -X POST http://localhost:8000/upload/ \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "PDF uploaded and indexed successfully.",
  "filename": "your_document.pdf",
  "chunks_indexed": 42
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Response:
```json
{
  "question": "What is the main topic of the document?",
  "answer": "The document discusses ...",
  "sources": [
    {
      "source_file": "your_document.pdf",
      "page": 3,
      "chunk_text": "The relevant passage from page 3 ..."
    }
  ]
}
```

The `sources` array lets you verify exactly which page and file each answer came from.

### Health check

```bash
curl http://localhost:8000/health/
```

---

## Configuration Reference

All settings are read from `.env` (see `.env.example`):

| Variable              | Default                                    | Description                          |
|-----------------------|--------------------------------------------|--------------------------------------|
| `GROQ_API_KEY`        | required                                   | Your Groq API key                    |
| `GROQ_MODEL_NAME`     | `llama-3.3-70b-versatile`                  | Groq model to use                    |
| `EMBEDDING_MODEL_NAME`| `sentence-transformers/all-mpnet-base-v2`  | Local HuggingFace embedding model    |
| `FAISS_INDEX_PATH`    | `faiss_index`                              | Directory where FAISS index is saved |
| `UPLOAD_DIR`          | `uploads`                                  | Directory where PDFs are saved       |
| `CHUNK_SIZE`          | `1000`                                     | Characters per text chunk            |
| `CHUNK_OVERLAP`       | `200`                                      | Overlap between consecutive chunks   |
| `RETRIEVER_K`         | `4`                                        | Number of chunks retrieved per query |

---

