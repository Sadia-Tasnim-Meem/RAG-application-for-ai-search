import pytest
from langchain_core.documents import Document

from app.retrieval.chain import format_docs_with_sources, parse_citations


def test_format_docs_with_sources():
    docs = [
        Document(page_content="Content of doc 1", metadata={"source": "doc1.md"}),
        Document(
            page_content="Content of doc 2",
            metadata={"source": "doc2.pdf", "page": 3},
        ),
    ]
    result = format_docs_with_sources(docs)
    assert "[1] Source: doc1.md" in result
    assert "[2] Source: doc2.pdf, Page 3" in result
    assert "Content of doc 1" in result
    assert "Content of doc 2" in result


def test_format_docs_with_header():
    docs = [
        Document(
            page_content="Section content",
            metadata={"source": "doc.md", "h1": "Introduction"},
        ),
    ]
    result = format_docs_with_sources(docs)
    assert "Section: Introduction" in result


def test_parse_citations():
    answer = "The sky is blue [1] and grass is green [2]."
    docs = [
        Document(
            page_content="The sky appears blue due to Rayleigh scattering.",
            metadata={"source": "sky.md"},
        ),
        Document(
            page_content="Grass contains chlorophyll which makes it green.",
            metadata={"source": "grass.pdf", "page": 5},
        ),
    ]
    cleaned_answer, citations = parse_citations(answer, docs)
    assert cleaned_answer == answer
    assert len(citations) == 2
    assert citations[0].chunk_id == 1
    assert citations[0].source == "sky.md"
    assert citations[1].chunk_id == 2
    assert citations[1].source == "grass.pdf"
    assert citations[1].page == 5


def test_parse_citations_with_no_references():
    answer = "I don't know the answer to that question."
    docs = [
        Document(page_content="Some content.", metadata={"source": "doc.md"}),
    ]
    cleaned_answer, citations = parse_citations(answer, docs)
    assert len(citations) == 0


def test_parse_citations_out_of_range_id():
    answer = "Some claim [5] is relevant."
    docs = [
        Document(page_content="Only one doc.", metadata={"source": "doc.md"}),
    ]
    cleaned_answer, citations = parse_citations(answer, docs)
    assert len(citations) == 0


@pytest.mark.slow
def test_query_with_citation_integration():
    from app.ingestion.store import create_vector_store
    from app.retrieval.chain import query_with_citation

    docs = [
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "geo.txt"},
        ),
    ]
    vector_store = create_vector_store(docs)
    result = query_with_citation("What is the capital of France?", vector_store)
    assert "Paris" in result.answer
    assert len(result.citations) > 0
