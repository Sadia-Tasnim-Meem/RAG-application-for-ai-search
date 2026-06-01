# RAG Application for AI Search

A **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangChain**, **Chroma DB**, and **Hugging Face**. Supports PDF, Markdown, and web pages with source citations.

---

## Pipeline

### Ingestion

```
Document (PDF / Markdown / Web URL)
    │
    ▼
Loader (PyMuPDF / pathlib / WebBaseLoader)
    │
    ▼
Splitter (RecursiveCharacterTextSplitter 800/100 + MarkdownHeaderTextSplitter)
    │
    ▼
Embedder (BGE-small-v1.5 → 384-dim vectors)
    │
    ▼
Chroma vector store (persisted to ./chroma_db)
```



### Query

```
Query
    │
    ▼
HybridRetriever (BM25 + vector, weighted 0.5/0.5 fusion, top-10)
    │
    ▼
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, top-4)
    │
    ▼
Qwen2.5-0.5B-Instruct via ChatHuggingFace
    │
    ▼
Answer with citations
```


---

## Tech Stack

- **FastAPI** — REST API backend
- **LangChain** — Document loading, splitting, retrieval, chat model interface
- **Chroma DB** — Vector database for embeddings
- **HuggingFace BGE-small** — Embedding model (local, free, 33 MB)
- **HuggingFace Qwen 2.5 0.5B-Instruct** — Answer generation (local, free, 900 MB)
- **Cross-encoder ms-marco-MiniLM-L-6-v2** — Reranker (local, free, 80 MB)
- **rank-bm25** — Sparse keyword retrieval
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

---

## Project Structure

```
app/
├── ingestion/
│   ├── loader.py      # PDF / Markdown / Web loaders
│   ├── splitter.py    # RecursiveCharacterTextSplitter + MarkdownHeaderTextSplitter
│   └── store.py       # Chroma persistence, get_all_documents() for BM25 corpus
├── retrieval/
│   ├── chain.py       # query_with_citation() — main orchestration
│   ├── config.py      # Loads prompts.yaml → ChatPromptTemplate
│   ├── prompts.yaml   # default + strict citation prompts
│   ├── reranker.py    # Cross-encoder reranker
│   └── retriever.py   # HybridRetriever (BM25 + vector fusion)
├── schemas.py         # Pydantic v2 models
├── main.py            # FastAPI entry (POST /ingest, POST /query)
└── cli.py             # CLI entry (ingest, query subcommands)
```

---

## Development

```bash
# Run all tests
uv run pytest -v

# Fast tests only (skip model downloads)
uv run pytest -v -m "not slow"

# Lint
uv run ruff check .

# Format
uv run ruff format .
```


