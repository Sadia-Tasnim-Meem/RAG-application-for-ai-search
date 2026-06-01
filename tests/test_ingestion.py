from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingestion.loader import load_markdown, load_pdf
from app.ingestion.splitter import split_documents


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "test.md"
    p.write_text("# Title\n\nThis is a test document.\n\n## Section 1\n\nSome content here.")
    return p


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    import fitz

    p = tmp_path / "test.pdf"
    doc = fitz.open()
    doc.new_page()
    page = doc[0]
    page.insert_text((50, 50), "Hello from PDF page 1")
    doc.new_page()
    page = doc[1]
    page.insert_text((50, 50), "Hello from PDF page 2")
    doc.save(str(p))
    doc.close()
    return p


def test_load_markdown(sample_md: Path):
    docs = load_markdown(sample_md)
    assert len(docs) == 1
    assert "test document" in docs[0].page_content
    assert docs[0].metadata["source"] == str(sample_md)


def test_load_pdf(sample_pdf: Path):
    docs = load_pdf(sample_pdf)
    assert len(docs) == 2
    assert "Hello from PDF page 1" in docs[0].page_content
    assert "Hello from PDF page 2" in docs[1].page_content
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2


def test_split_markdown(sample_md: Path):
    docs = load_markdown(sample_md)
    chunks = split_documents(docs)
    assert len(chunks) > 0


def test_split_plain_text():
    docs = [Document(page_content="Word. " * 200, metadata={"source": "test.txt"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_split_pdf(sample_pdf: Path):
    docs = load_pdf(sample_pdf)
    chunks = split_documents(docs)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "page" in chunk.metadata
