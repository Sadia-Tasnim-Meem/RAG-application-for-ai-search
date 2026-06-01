# RAG Application for AI Search

A **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangChain**, **Chroma DB**, and **Hugging Face**. Supports PDF, Markdown, and web pages with source citations.

---

## What it Does

- Ingest documents from PDFs, markdown files, or web URLs
- Split into chunks 
- Embed with BGE-small and store in Chroma vector DB
- Retrieve top-k relevant chunks for a query
- Generate an answer with source citations using a HuggingFace model

---

## Tech Stack

- **FastAPI** — REST API backend
- **LangChain** — Document loading, splitting, retrieval 
- **Chroma DB** — Vector database for embeddings
- **HuggingFace BGE-small** — Embedding model (local, free)
- **HuggingFace Qwen 2.5 0.5B-Instruct** — Answer generation (local, free)
- **PyMuPDF** — PDF text extraction
- **uv** — Python package manager 

---


## Setup

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Start the API
```bash
uv run uvicorn app.main:app --reload --port 8000
```

---

## CLI Usage

```bash
# Ingest a file
uv run python -m app.cli ingest file --path <file_path>

# Ingest from web
uv run python -m app.cli ingest web --urls https://en.wikipedia.org/wiki/RAG

# Query
uv run python -m app.cli query "<query>"
```


## Development

```bash
# Run tests
uv run pytest -v

# Lint
uv run ruff check .

# Format
uv run ruff format .
```
