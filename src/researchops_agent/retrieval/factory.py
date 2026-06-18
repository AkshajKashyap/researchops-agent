from researchops_agent.retrieval.base import BaseRetriever
from researchops_agent.retrieval.embedding import EmbeddingRetriever
from researchops_agent.retrieval.tfidf import TfidfRetriever


def build_retriever(kind: str) -> BaseRetriever:
    normalized_kind = kind.strip().lower()
    if normalized_kind == "tfidf":
        return TfidfRetriever()
    if normalized_kind == "embedding":
        return EmbeddingRetriever()
    raise ValueError(f"Unsupported retriever kind: {kind}")
