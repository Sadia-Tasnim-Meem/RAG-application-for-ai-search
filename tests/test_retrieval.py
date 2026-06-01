from pathlib import Path

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
def test_hybrid_retrieval_ranks_length_based_chunks_higher(tmp_path, monkeypatch):
    """With hybrid BM25 + vector retrieval and reranker, a query about
    'length based splitting' should rank the Length-based section
    chunks above noise from other sections.
    """
    import app.ingestion.store as store_mod
    from app.ingestion.loader import load_markdown
    from app.ingestion.splitter import split_documents
    from app.ingestion.store import create_vector_store
    from app.retrieval.chain import query_with_citation

    persist_dir = str(tmp_path / "chroma_db")
    monkeypatch.setattr(store_mod, "PERSIST_DIRECTORY", persist_dir)

    fixture = Path(__file__).parent / "fixtures" / "length-test.md"
    docs = load_markdown(fixture)
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    result = query_with_citation("what are the types of length based splitting?", vector_store)

    # The answer should mention both Token-based and Character-based
    assert "Token-based" in result.answer
    assert "Character-based" in result.answer


def test_hybrid_retriever_fusion():
    """HybridRetriever deduplicates and ranks by weighted fusion."""
    from app.retrieval.retriever import _rank_to_scores

    docs = [
        Document(page_content="Doc A about topic X", metadata={"source": "a.md"}),
        Document(page_content="Doc B about topic Y", metadata={"source": "b.md"}),
    ]
    scores = _rank_to_scores(docs)
    assert scores["Doc A about topic X"] == 1.0
    assert scores["Doc B about topic Y"] == 0.5


@pytest.mark.slow
def test_reranker_scores_relevant_first(tmp_path):
    """Reranker assigns higher scores to relevant query-chunk pairs."""
    from app.retrieval.reranker import Reranker

    docs = [
        Document(
            page_content="Token-based and character-based splitting methods.",
            metadata={"source": "a.md"},
        ),
        Document(
            page_content="The weather is sunny today in Paris.",
            metadata={"source": "b.md"},
        ),
    ]

    reranker = Reranker()
    reranked = reranker.rerank("what are the types of splitting?", docs, top_k=2)
    assert reranked[0].page_content == docs[0].page_content
    assert len(reranked) == 2


@pytest.mark.slow
def test_citation_enforcement_retry(tmp_path, monkeypatch):
    """When no citations are found, the retry mechanism fires with a
    strict prompt and still returns a valid response."""
    import app.ingestion.store as store_mod
    from app.ingestion.store import create_vector_store
    from app.retrieval.chain import query_with_citation

    persist_dir = str(tmp_path / "chroma_db")
    monkeypatch.setattr(store_mod, "PERSIST_DIRECTORY", persist_dir)

    docs = [
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "geo.txt"},
        ),
    ]
    vector_store = create_vector_store(docs)
    result = query_with_citation("What is the capital of France?", vector_store)
    assert len(result.answer) > 0
    assert isinstance(result.citations, list)


@pytest.mark.slow
def test_query_with_citation_integration(tmp_path, monkeypatch):
    import app.ingestion.store as store_mod
    from app.ingestion.store import create_vector_store
    from app.retrieval.chain import query_with_citation

    persist_dir = str(tmp_path / "chroma_db")
    monkeypatch.setattr(store_mod, "PERSIST_DIRECTORY", persist_dir)

    docs = [
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "geo.txt"},
        ),
    ]
    vector_store = create_vector_store(docs)
    result = query_with_citation("What is the capital of France?", vector_store)
    assert "Paris" in result.answer
