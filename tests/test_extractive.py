import pytest

from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.schemas.answer import EvidenceItem, EvidencePack


def _pack(items: list[EvidenceItem]) -> EvidencePack:
    return EvidencePack(query="What does retrieval use?", items=items)


def _item(text: str = "Retrieval uses TF-IDF over local chunks. Other text.") -> EvidenceItem:
    return EvidenceItem(
        chunk_id="chunk-0",
        citation="data/raw/example.md#chunk=0",
        score=0.9,
        text=text,
    )


def test_extractive_answer_returns_non_abstained_answer_when_evidence_exists() -> None:
    answer = answer_from_evidence(_pack([_item()]))

    assert answer.abstained is False
    assert answer.answer == "Retrieval uses TF-IDF over local chunks."


def test_extractive_answer_includes_citation() -> None:
    answer = answer_from_evidence(_pack([_item()]))

    assert answer.citations == ["data/raw/example.md#chunk=0"]


def test_extractive_answer_abstains_when_evidence_is_empty() -> None:
    answer = answer_from_evidence(_pack([]))

    assert answer.abstained is True
    assert answer.citations == []
    assert "not have enough retrieved evidence" in answer.answer
    assert answer.reason == "Insufficient retrieved evidence."


def test_extractive_answer_rejects_invalid_min_items() -> None:
    with pytest.raises(ValueError, match="min_items must be positive"):
        answer_from_evidence(_pack([_item()]), min_items=0)
