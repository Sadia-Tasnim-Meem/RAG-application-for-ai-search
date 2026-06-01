from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(documents: list[Document]) -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vector_store


def get_vector_store() -> Chroma | None:
    if not Path(PERSIST_DIRECTORY).exists():
        return None
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
