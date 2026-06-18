from researchops_agent.llm.validation import validate_grounded_answer
from researchops_agent.schemas.answer import EvidenceItem, EvidencePack
from researchops_agent.schemas.llm import GroundedLLMAnswer


def _pack() -> EvidencePack:
    return EvidencePack(
        query="query",
        items=[
            EvidenceItem(
                chunk_id="chunk-0",
                citation="paper.md#chunk=0",
                score=0.9,
                text="Evidence text",
            )
        ],
    )


def test_validation_accepts_citations_present_in_evidence() -> None:
    answer = GroundedLLMAnswer(
        query="query",
        answer="Evidence text",
        citations=["paper.md#chunk=0"],
    )

    validated = validate_grounded_answer(answer, _pack())

    assert validated.abstained is False


def test_validation_abstains_if_citation_not_in_evidence() -> None:
    answer = GroundedLLMAnswer(
        query="query",
        answer="Evidence text",
        citations=["other.md#chunk=0"],
    )

    validated = validate_grounded_answer(answer, _pack())

    assert validated.abstained is True
    assert "outside retrieved evidence" in validated.reason


def test_validation_abstains_if_non_abstained_answer_has_no_citation() -> None:
    answer = GroundedLLMAnswer(query="query", answer="Evidence text", citations=[])

    validated = validate_grounded_answer(answer, _pack())

    assert validated.abstained is True
    assert "did not include citations" in validated.reason


def test_validation_allows_abstention_with_no_citations() -> None:
    answer = GroundedLLMAnswer(
        query="query",
        answer="No evidence",
        citations=[],
        abstained=True,
    )

    validated = validate_grounded_answer(answer, _pack())

    assert validated.abstained is True
    assert validated.citations == []
