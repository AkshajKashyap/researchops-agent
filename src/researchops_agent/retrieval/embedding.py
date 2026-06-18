from typing import Any

import numpy as np

from researchops_agent.schemas.document import DocumentChunk
from researchops_agent.schemas.retrieval import RetrievedChunk, RetrievalResult

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever:
    def __init__(
        self,
        model: object | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model = model
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._chunks: list[DocumentChunk] = []
        self._matrix: np.ndarray | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise ValueError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "The first real embedding run may require downloading the model."
                ) from exc
        return self._model

    def fit(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot fit retriever without chunks")

        self._chunks = chunks
        embeddings = np.asarray(
            self._get_model().encode([chunk.text for chunk in chunks]), dtype=float
        )
        self._matrix = self._prepare_embeddings(embeddings)

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        if self._matrix is None:
            raise ValueError("EmbeddingRetriever must be fitted before search")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = np.asarray(self._get_model().encode([query]), dtype=float)
        query_matrix = self._prepare_embeddings(query_embedding)
        scores = self._cosine_scores(query_matrix[0])
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:top_k]

        results = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source.path,
                source_type=chunk.source.source_type,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=float(score),
            )
            for index, score in ranked
            for chunk in [self._chunks[index]]
        ]

        return RetrievalResult(query=query, results=results)

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self.normalize_embeddings:
            return self._normalize(embeddings)
        return embeddings

    def _cosine_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        if self._matrix is None:
            raise ValueError("EmbeddingRetriever must be fitted before search")
        if self.normalize_embeddings:
            return self._matrix @ query_embedding

        matrix = self._normalize(self._matrix)
        query = self._normalize(query_embedding.reshape(1, -1))[0]
        return matrix @ query

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return embeddings / safe_norms
