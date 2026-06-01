from fastapi import FastAPI, HTTPException

from app.ingestion.loader import load_document
from app.ingestion.splitter import split_documents
from app.ingestion.store import create_vector_store, get_vector_store
from app.retrieval.chain import query_with_citation
from app.schemas import IngestRequest, QueryRequest, QueryResponse

app = FastAPI(title="RAG Application", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "RAG Application with Citation"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        documents = load_document(req.source_type, file_path=req.file_path, urls=req.urls)
        chunks = split_documents(documents)
        create_vector_store(chunks)
        return {"status": "success", "chunks_created": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    vector_store = get_vector_store()
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Call /ingest first.",
        )
    return query_with_citation(req.question, vector_store)
