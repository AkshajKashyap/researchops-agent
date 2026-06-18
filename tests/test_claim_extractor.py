import pytest

from researchops_agent.agents.claim_extractor import extract_experiment_claims
from researchops_agent.schemas.answer import EvidenceItem, EvidencePack


def _pack(text: str) -> EvidencePack:
    return EvidencePack(
        query="What experiment is described?",
        items=[
            EvidenceItem(
                chunk_id="chunk-0",
                citation="data/raw/paper.md#chunk=0",
                score=0.8,
                text=text,
            )
        ],
    )


def test_extracts_claim_from_experiment_language() -> None:
    pack = _pack("We compare Ridge and LSTM on the MIMIC dataset.")

    claims = extract_experiment_claims(pack)

    assert len(claims) == 1
    assert claims[0].main_claim == "We compare Ridge and LSTM on the MIMIC dataset."
    assert claims[0].models == ["Ridge", "LSTM"]
    assert claims[0].dataset == "MIMIC"


def test_returns_empty_list_when_no_experiment_language_exists() -> None:
    pack = _pack("This document introduces the project structure and motivation.")

    assert extract_experiment_claims(pack) == []


def test_extracts_obvious_metrics() -> None:
    pack = _pack("Results show the model achieves accuracy, F1, RMSE, and AUC gains.")

    claims = extract_experiment_claims(pack)

    assert claims[0].metrics == ["accuracy", "F1", "RMSE", "AUC"]


def test_preserves_evidence_citation() -> None:
    pack = _pack("Results show Ridge outperforms the baseline on accuracy.")

    claims = extract_experiment_claims(pack)

    assert claims[0].evidence_citations == ["data/raw/paper.md#chunk=0"]


def test_rejects_invalid_max_claims() -> None:
    with pytest.raises(ValueError, match="max_claims must be positive"):
        extract_experiment_claims(_pack("We evaluate Ridge."), max_claims=0)
