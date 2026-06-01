from typing import Optional

from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: int
    source: str
    page: Optional[int] = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


class IngestRequest(BaseModel):
    source_type: str
    file_path: Optional[str] = None
    urls: Optional[list[str]] = None


class QueryRequest(BaseModel):
    question: str
