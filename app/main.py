from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import generator as generator

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.post("/query")
def query(req: QueryRequest):
    query_embedding = model.encode([req.question])
    D,I = index.search(np.array(query_embedding), k=2)
    best_chunk  = docs[I[0][0]]
    answer = generator.generate_answer(req.question, best_chunk)

    return {
        "question" : req.question,
        "retrieved_context" : best_chunk,
        "generated_answer":answer
    }


docs = open("../data/docs.txt").read().split("\n\n")
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)

index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
index.add(doc_embeddings)