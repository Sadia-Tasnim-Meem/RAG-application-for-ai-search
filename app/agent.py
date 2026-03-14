from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from app.ingestion import get_vector_store

def get_retriever(vector_store, k=2):
    """Get retriever from vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})

def setup_qa_chain(vector_store):
    """Setup QA chain with retrieval and generation."""
    retriever = get_retriever(vector_store)
    
    prompt = ChatPromptTemplate.from_template("""Answer the question based on the provided context.

Context: {context}

Question: {question}

Answer:""")
    
    from langchain_community.llms import HuggingFaceHub
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnableIdentity()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

class RunnableIdentity:
    def invoke(self, input, config=None):
        return input

def query_with_agent(question: str, vector_store=None):
    """Query the RAG system."""
    if vector_store is None:
        vector_store = get_vector_store()
    
    if vector_store is None:
        return {"error": "No vector store found. Please ingest data first."}
    
    chain = setup_qa_chain(vector_store)
    answer = chain.invoke(question)
    
    docs = get_similar_documents(question, vector_store, k=2)
    
    return {
        "question": question,
        "answer": answer,
        "context": docs
    }

def get_similar_documents(question: str, vector_store=None, k=2):
    """Get similar documents for a query."""
    if vector_store is None:
        vector_store = get_vector_store()
    
    if vector_store is None:
        return []
    
    retriever = get_retriever(vector_store, k=k)
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]
