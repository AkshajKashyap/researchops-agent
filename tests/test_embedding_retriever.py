import pytest

from researchops_agent.retrieval.embedding import EmbeddingRetriever
from researchops_agent.schemas.document import DocumentChunk, DocumentSource


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        terms = ["ridge", "lstm", "transformer", "rmse", "mae"]
        return [
            [float(text.lower().count(term)) for term in terms]
            for text in texts
        ]


def _chunk(
    text: str,
    *,
    chunk_id: str = "paper.md::page:none::chunk:0",
    page_number: int | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    source = DocumentSource(path="paper.md", source_type="markdown")
    return DocumentChunk(
        chunk_id=chunk_id,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_embedding_retriever_fit_and_search_with_fake_model() -> None:
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())
    chunks = [
        _chunk("Ridge model with RMSE"),
        _chunk("LSTM model with MAE", chunk_id="paper.md::page:none::chunk:1", chunk_index=1),
    ]

    retriever.fit(chunks)
    result = retriever.search("Ridge RMSE")

    assert result.query == "Ridge RMSE"
    assert result.results[0].chunk_id == chunks[0].chunk_id


def test_embedding_retriever_higher_score_result_appears_first() -> None:
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())
    retriever.fit(
        [
            _chunk("LSTM model with MAE"),
            _chunk("Ridge model with RMSE", chunk_id="paper.md::page:none::chunk:1", chunk_index=1),
        ]
    )

    result = retriever.search("Ridge RMSE")

    assert result.results[0].chunk_index == 1
    assert result.results[0].score > result.results[1].score


def test_embedding_retriever_preserves_metadata() -> None:
    chunk = _chunk(
        "Transformer model with MAE",
        chunk_id="paper.pdf::page:2::chunk:3",
        page_number=2,
        chunk_index=3,
    )
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())
    retriever.fit([chunk])

    result = retriever.search("Transformer")
    retrieved = result.results[0]

    assert retrieved.chunk_id == "paper.pdf::page:2::chunk:3"
    assert retrieved.source_path == "paper.md"
    assert retrieved.source_type == "markdown"
    assert retrieved.page_number == 2
    assert retrieved.chunk_index == 3
    assert retrieved.text == "Transformer model with MAE"


def test_embedding_retriever_search_before_fit_raises_value_error() -> None:
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="fitted before search"):
        retriever.search("Ridge")


def test_embedding_retriever_empty_query_raises_value_error() -> None:
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())
    retriever.fit([_chunk("Ridge model")])

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search("   ")


def test_embedding_retriever_invalid_top_k_raises_value_error() -> None:
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())
    retriever.fit([_chunk("Ridge model")])

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("Ridge", top_k=0)


def test_embedding_retriever_top_k_limit_is_respected() -> None:
    retriever = EmbeddingRetriever(model=FakeEmbeddingModel())
    retriever.fit(
        [
            _chunk("Ridge model"),
            _chunk("LSTM model", chunk_id="paper.md::page:none::chunk:1", chunk_index=1),
            _chunk(
                "Transformer model",
                chunk_id="paper.md::page:none::chunk:2",
                chunk_index=2,
            ),
        ]
    )

    result = retriever.search("model", top_k=2)

    assert len(result.results) == 2
