from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.ingestion import ingest_data, get_vector_store
from app.agent import query_with_agent, get_similar_documents

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    source_type: str
    source_path: Optional[str] = None
    urls: Optional[List[str]] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200

@app.get("/")
async def root():
    return {"message": "RAG Application with LangChain"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        vector_store, chunks = ingest_data(
            source_type=req.source_type,
            source_path=req.source_path,
            urls=req.urls
        )
        return {
            "status": "success",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(req: QueryRequest):
    vector_store = get_vector_store()
    
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No data ingested. Please call /ingest first."
        )
    
    result = query_with_agent(req.question, vector_store)
    return result

@app.get("/similar/{question}")
def similar(question: str, k: int = 2):
    vector_store = get_vector_store()
    
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No data ingested. Please call /ingest first."
        )
    
    docs = get_similar_documents(question, vector_store, k)
    return {"documents": docs}
