import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

PERSIST_DIRECTORY = "./chroma_db"

def load_documents_from_web(urls: list[str]):
    """Load documents from web URLs using WebBaseLoader."""
    loader = WebBaseLoader(web_paths=urls)
    documents = loader.load()
    return documents

def load_documents_from_file(file_path: str):
    """Load documents from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    from langchain_core.documents import Document
    doc = Document(page_content=content, metadata={"source": file_path})
    return [doc]

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """Create Chroma vector store from documents."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(PERSIST_DIRECTORY):
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
    
    return vector_store

def ingest_data(source_type="file", source_path=None, urls=None):
    """Main ingestion function."""
    if source_type == "web" and urls:
        documents = load_documents_from_web(urls)
    elif source_type == "file" and source_path:
        documents = load_documents_from_file(source_path)
    else:
        raise ValueError("Invalid source type or missing source")
    
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    
    return vector_store, chunks

def get_vector_store():
    """Get existing vector store or create new one."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return None
