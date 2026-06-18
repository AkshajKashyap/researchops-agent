import pytest

from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.schemas.retrieval import RetrievedChunk, RetrievalResult


def _retrieved(
    *,
    chunk_id: str = "chunk-0",
    source_path: str = "data/raw/paper.pdf",
    source_type: str = "pdf",
    page_number: int | None = 2,
    chunk_index: int = 0,
    text: str = "Evidence text.",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path=source_path,
        source_type=source_type,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def test_evidence_pack_preserves_query() -> None:
    result = RetrievalResult(query="what is retrieval", results=[_retrieved()])

    pack = build_evidence_pack(result)

    assert pack.query == "what is retrieval"


def test_evidence_pack_creates_citation_strings() -> None:
    result = RetrievalResult(query="query", results=[_retrieved()])

    pack = build_evidence_pack(result)

    assert pack.items[0].citation == "data/raw/paper.pdf#page=2&chunk=0"


def test_evidence_pack_filters_by_min_score() -> None:
    result = RetrievalResult(
        query="query",
        results=[
            _retrieved(chunk_id="high", score=0.9),
            _retrieved(chunk_id="low", score=0.1, chunk_index=1),
        ],
    )

    pack = build_evidence_pack(result, min_score=0.5)

    assert [item.chunk_id for item in pack.items] == ["high"]


def test_evidence_pack_rejects_negative_min_score() -> None:
    result = RetrievalResult(query="query", results=[])

    with pytest.raises(ValueError, match="min_score must be non-negative"):
        build_evidence_pack(result, min_score=-0.1)
