from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.llm import GroundedLLMAnswer


def _abstain(answer: GroundedLLMAnswer, reason: str) -> GroundedLLMAnswer:
    return GroundedLLMAnswer(
        query=answer.query,
        answer="I do not have enough retrieved evidence to answer this question.",
        citations=[],
        abstained=True,
        reason=reason,
        provider=answer.provider,
        model=answer.model,
    )


def validate_grounded_answer(
    answer: GroundedLLMAnswer, evidence: EvidencePack
) -> GroundedLLMAnswer:
    if answer.abstained:
        return answer

    if not answer.answer.strip():
        return _abstain(answer, "LLM answer was empty.")
    if not answer.citations:
        return _abstain(answer, "LLM answer did not include citations.")

    allowed_citations = {item.citation for item in evidence.items}
    invalid_citations = [
        citation for citation in answer.citations if citation not in allowed_citations
    ]
    if invalid_citations:
        return _abstain(answer, "LLM answer cited sources outside retrieved evidence.")

    return answer
