from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from researchops_agent.schemas.document import DocumentChunk
from researchops_agent.schemas.retrieval import RetrievedChunk, RetrievalResult


class TfidfRetriever:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._chunks: list[DocumentChunk] = []
        self._matrix = None
        self._is_fitted = False

    def fit(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot fit retriever without chunks")

        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in chunks)
        self._is_fitted = True

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self._is_fitted or self._matrix is None:
            raise ValueError("TfidfRetriever must be fitted before search")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
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
