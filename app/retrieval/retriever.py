from collections import defaultdict

from langchain_chroma import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(
        self,
        vector_store: Chroma,
        all_docs: list[Document],
        k: int = 10,
        weights: tuple[float, float] = (0.5, 0.5),
    ):
        self.bm25 = BM25Retriever.from_documents(all_docs, k=k)
        self.vector = vector_store.as_retriever(search_kwargs={"k": k})
        self.k = k
        self.weights = weights

    def invoke(self, query: str) -> list[Document]:
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vector.invoke(query)

        bm25_scores = _rank_to_scores(bm25_docs)
        vector_scores = _rank_to_scores(vector_docs)

        combined = defaultdict(float)
        for doc in bm25_docs:
            doc_key = doc.page_content
            combined[doc_key] += bm25_scores[doc_key] * self.weights[0]
        for doc in vector_docs:
            doc_key = doc.page_content
            combined[doc_key] += vector_scores[doc_key] * self.weights[1]

        seen = set()
        ranked = []
        for doc in bm25_docs + vector_docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                ranked.append((doc, combined.get(key, 0.0)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.k]]


def _rank_to_scores(docs: list[Document]) -> dict[str, float]:
    n = len(docs)
    if n == 0:
        return {}
    return {doc.page_content: (n - i) / n for i, doc in enumerate(docs)}
