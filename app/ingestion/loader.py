from pathlib import Path

import fitz
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    doc = fitz.open(path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_num + 1},
                )
            )
    doc.close()
    return docs


def load_markdown(path: str | Path) -> list[Document]:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return [Document(page_content=content, metadata={"source": str(path)})]


def load_web(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(web_paths=urls)
    return loader.load()


def load_document(
    source_type: str,
    file_path: str | None = None,
    urls: list[str] | None = None,
) -> list[Document]:
    if source_type == "web":
        if not urls:
            raise ValueError("URLs required for web source type")
        return load_web(urls)

    if not file_path:
        raise ValueError("file_path required for file source type")

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".md", ".markdown"):
        return load_markdown(path)
    else:
        return load_markdown(path)
