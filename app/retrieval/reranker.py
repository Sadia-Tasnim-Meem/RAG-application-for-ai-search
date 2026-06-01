from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[Document], top_k: int = 4) -> list[Document]:
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored[:top_k]]
