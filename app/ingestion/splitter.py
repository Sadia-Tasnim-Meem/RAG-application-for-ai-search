from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks: list[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source.endswith((".md", ".markdown")):
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "h1"),
                    ("##", "h2"),
                    ("###", "h3"),
                ],
            )
            header_chunks = header_splitter.split_text(doc.page_content)
            for hc in header_chunks:
                for key, val in doc.metadata.items():
                    if key not in hc.metadata:
                        hc.metadata[key] = val
            chunks = text_splitter.split_documents(header_chunks)
            all_chunks.extend(chunks)
        else:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

    return all_chunks
