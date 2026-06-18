from typing import Protocol

from researchops_agent.schemas.document import DocumentChunk
from researchops_agent.schemas.retrieval import RetrievalResult


class BaseRetriever(Protocol):
    def fit(self, chunks: list[DocumentChunk]) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        ...
