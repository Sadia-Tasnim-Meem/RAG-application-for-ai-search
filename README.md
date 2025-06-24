# 🧠 RAG application for AI search 

This is a **minimal Retrieval-Augmented Generation (RAG)** API with a simple frontend, built using **FastAPI**, **FAISS**, **SentenceTransformers**, and **Hugging Face Transformers**.

This project was built as part of my hands-on practice while learning and exploring **agentic AI** and **RAG architectures**. The goal was to build something functional and complete and gradually expand it as I deepen my understanding.

---

## 🔍 What it Does

- Accepts a question via a REST API or a Streamlit-based frontend
- Retrieves the most relevant content chunk from a small knowledge base
- Optionally generates an answer using Hugging Face’s `flan-t5-small` model

> 📝 The current knowledge base is filled with AI-generated demo content related to modern AI tools and concepts (e.g., FAISS, LangChain, Transformers, etc.)

---

## 🚀 Tech Stack

- **FastAPI** — lightweight backend API
- **SentenceTransformers** — for converting docs & queries into embeddings
- **FAISS** — vector similarity search
- **Hugging Face Transformers** — answer generation (local model)
- **Streamlit** — frontend interface

---

## 📁 Usage

### 1. Start the API backend
```bash
uvicorn main:app --reload
```

### **2. Start the Streamlit app**
```bash
streamlit run frontend.py
```
Then open your browser at: http://localhost:8501

