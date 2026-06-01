import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

from app.retrieval.config import load_prompt
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import HybridRetriever
from app.schemas import Citation, QueryResponse

HYBRID_K = 10
RERANK_TOP_K = 4
HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_retriever(vector_store: Chroma):
    from app.ingestion.store import get_all_documents

    all_docs = get_all_documents()
    return HybridRetriever(vector_store, all_docs, k=HYBRID_K)


def get_hf_llm():
    hf_pipeline = pipeline(
        "text-generation",
        model=HF_MODEL,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return ChatHuggingFace(llm=llm)


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


def _clean_chat_output(text: str) -> str:
    idx = text.find("<|im_start|>assistant")
    if idx != -1:
        text = text[idx + len("<|im_start|>assistant") :]
        text = text.replace("<|im_end|>", "").strip()
    return text


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

    retriever = build_retriever(vector_store)
    docs = retriever.invoke(question)

    if not docs:
        return QueryResponse(
            question=question,
            answer="No relevant documents found for your question.",
            citations=[],
        )

    reranker = Reranker()
    docs = reranker.rerank(question, docs, top_k=RERANK_TOP_K)

    context = format_docs_with_sources(docs)
    llm = get_hf_llm()

    prompt = load_prompt("default")
    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = chain.invoke({"context": context, "question": question})
    answer_text = _clean_chat_output(response.content)
    answer_text, citations = parse_citations(answer_text, docs)

    if not citations:
        strict_prompt = load_prompt("strict")
        strict_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | strict_prompt
            | llm
        )
        response = strict_chain.invoke({"context": context, "question": question})
        answer_text = _clean_chat_output(response.content)
        answer_text, citations = parse_citations(answer_text, docs)

    return QueryResponse(
        question=question,
        answer=answer_text,
        citations=citations,
    )
