# RAG Application for AI Search

A **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, **LangChain**, **Chroma DB**, and **Hugging Face Transformers**.

---

## What it Does

- Load documents from web URLs (using LangChain WebBaseLoader) or local files
- Split documents into chunks (using LangChain RecursiveCharacterTextSplitter)
- Store embeddings in Chroma DB for vector similarity search
- Answer questions using a RAG chain with retrieval and generation

---

## Tech Stack

- **FastAPI** — REST API backend
- **LangChain** — Data loading, text splitting, and RAG chain
- **Chroma DB** — Vector database for embeddings
- **SentenceTransformers** — Embedding model (all-MiniLM-L6-v2)
- **Hugging Face Transformers** — Answer generation (flan-t5-small)
- **Streamlit** — Frontend interface

---

## Project Structure

```
app/
├── main.py       # FastAPI endpoints
├── ingestion.py  # Data loading, splitting, and Chroma DB storage
├── agent.py      # RAG chain for retrieval and generation
├── generator.py  # Original generator (legacy)
└── frontend.py   # Streamlit frontend
```

---

## Setup

### 1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the API backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Start the Streamlit frontend (optional)
```bash
streamlit run app/frontend.py
```

---

## API Endpoints

### POST /ingest
Ingest data from file or web URLs.

**Request (from file):**
```json
{
  "source_type": "file",
  "source_path": "data/docs.txt"
}
```

**Request (from web):**
```json
{
  "source_type": "web",
  "urls": ["https://en.wikipedia.org/wiki/AI"]
}
```

### POST /query
Ask a question to the RAG system.

```json
{
  "question": "What is LangChain?"
}
```

### GET /similar/{question}
Get similar documents for a query.

```
/similar/What is AI?k=2
```

---

## Example Usage

```bash
# Ingest data
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "file", "source_path": "data/docs.txt"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```
