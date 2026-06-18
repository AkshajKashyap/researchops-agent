import pytest

from researchops_agent.retrieval.citations import format_citation
from researchops_agent.retrieval.tfidf import TfidfRetriever
from researchops_agent.schemas.document import DocumentChunk, DocumentSource
from researchops_agent.schemas.retrieval import RetrievedChunk


def _chunk(
    text: str,
    *,
    chunk_id: str = "paper.txt::page:none::chunk:0",
    path: str = "paper.txt",
    source_type: str = "text",
    page_number: int | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    source = DocumentSource(path=path, source_type=source_type)
    return DocumentChunk(
        chunk_id=chunk_id,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_fit_and_search_returns_results() -> None:
    retriever = TfidfRetriever()
    chunks = [
        _chunk("transformer attention models for retrieval"),
        _chunk("gardening notes about soil and compost", chunk_id="notes::chunk:1", chunk_index=1),
    ]

    retriever.fit(chunks)
    result = retriever.search("attention retrieval")

    assert result.query == "attention retrieval"
    assert len(result.results) == 2
    assert result.results[0].chunk_id == chunks[0].chunk_id


def test_higher_score_result_appears_first_for_relevant_query() -> None:
    retriever = TfidfRetriever()
    retriever.fit(
        [
            _chunk("climate policy and carbon accounting"),
            _chunk("neural retrieval uses transformer attention", chunk_id="paper::chunk:1", chunk_index=1),
        ]
    )

    result = retriever.search("transformer attention")

    assert result.results[0].chunk_index == 1
    assert result.results[0].score > result.results[1].score


def test_search_before_fit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fitted before search"):
        TfidfRetriever().search("attention")


def test_empty_query_raises_value_error() -> None:
    retriever = TfidfRetriever()
    retriever.fit([_chunk("attention retrieval")])

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search("   ")


def test_invalid_top_k_raises_value_error() -> None:
    retriever = TfidfRetriever()
    retriever.fit([_chunk("attention retrieval")])

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("attention", top_k=0)


def test_top_k_limit_is_respected() -> None:
    retriever = TfidfRetriever()
    retriever.fit(
        [
            _chunk("alpha beta"),
            _chunk("alpha gamma", chunk_id="paper::chunk:1", chunk_index=1),
            _chunk("alpha delta", chunk_id="paper::chunk:2", chunk_index=2),
        ]
    )

    result = retriever.search("alpha", top_k=2)

    assert len(result.results) == 2


def test_metadata_is_preserved_in_retrieved_results() -> None:
    chunk = _chunk(
        "attention retrieval",
        chunk_id="data/raw/paper.pdf::page:4::chunk:2",
        path="data/raw/paper.pdf",
        source_type="pdf",
        page_number=4,
        chunk_index=2,
    )
    retriever = TfidfRetriever()
    retriever.fit([chunk])

    result = retriever.search("attention")
    retrieved = result.results[0]

    assert retrieved.chunk_id == chunk.chunk_id
    assert retrieved.source_path == "data/raw/paper.pdf"
    assert retrieved.source_type == "pdf"
    assert retrieved.page_number == 4
    assert retrieved.chunk_index == 2
    assert retrieved.text == "attention retrieval"


def test_citation_formatting_with_page_number() -> None:
    result = RetrievedChunk(
        chunk_id="chunk-id",
        source_path="data/raw/paper.pdf",
        source_type="pdf",
        page_number=3,
        chunk_index=2,
        text="content",
        score=0.9,
    )

    assert format_citation(result) == "data/raw/paper.pdf#page=3&chunk=2"


def test_citation_formatting_without_page_number() -> None:
    result = RetrievedChunk(
        chunk_id="chunk-id",
        source_path="data/raw/notes.md",
        source_type="markdown",
        chunk_index=0,
        text="content",
        score=0.9,
    )

    assert format_citation(result) == "data/raw/notes.md#chunk=0"
