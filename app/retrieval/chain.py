import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from app.schemas import Citation, QueryResponse

DEFAULT_K = 4
HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_retriever(vector_store: Chroma, k: int = DEFAULT_K):
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_hf_llm():
    hf_pipeline = pipeline(
        "text-generation",
        model=HF_MODEL,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


def format_docs_with_sources(docs: list[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        header_info = ""
        for key in ["h1", "h2", "h3"]:
            if key in doc.metadata and doc.metadata[key]:
                header_info = f" (Section: {doc.metadata[key]})"
                break
        page_info = f", Page {page}" if page else ""
        formatted.append(f"[{i}] Source: {source}{page_info}{header_info}\n{doc.page_content}")
    return "\n\n".join(formatted)


CITATION_PROMPT = PromptTemplate.from_template(
    """You are an AI assistant answering questions based on provided documents.

Documents:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the provided documents.
2. For each factual claim, cite the source document number in brackets like [1], [2].
3. If the documents don't contain enough information, say so.
4. Be concise but thorough.

Answer:"""
)


def parse_citations(answer: str, docs: list[Document]) -> tuple[str, list[Citation]]:
    citation_ids = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        citation_ids.add(int(match.group(1)))

    citations = []
    for cid in sorted(citation_ids):
        idx = cid - 1
        if 0 <= idx < len(docs):
            doc = docs[idx]
            text_snippet = doc.page_content[:200]
            if len(doc.page_content) > 200:
                text_snippet += "..."
            citations.append(
                Citation(
                    chunk_id=cid,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    text=text_snippet,
                )
            )

    return answer, citations


def query_with_citation(question: str, vector_store: Chroma | None = None) -> QueryResponse:
    if vector_store is None:
        from app.ingestion.store import get_vector_store

        vector_store = get_vector_store()

    if vector_store is None:
        return QueryResponse(
            question=question,
            answer="No documents have been ingested yet. Please ingest documents first.",
            citations=[],
        )

    retriever = get_retriever(vector_store)
    docs = retriever.invoke(question)

    if not docs:
        return QueryResponse(
            question=question,
            answer="No relevant documents found for your question.",
            citations=[],
        )

    context = format_docs_with_sources(docs)
    llm = get_hf_llm()

    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | CITATION_PROMPT
        | llm
    )

    answer_text = chain.invoke({"context": context, "question": question})

    answer_text, citations = parse_citations(answer_text, docs)

    return QueryResponse(
        question=question,
        answer=answer_text,
        citations=citations,
    )
