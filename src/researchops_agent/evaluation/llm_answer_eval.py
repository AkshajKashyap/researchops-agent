from researchops_agent.agents.llm_grounded import answer_from_evidence_with_llm
from researchops_agent.pipeline import build_evidence_from_document
from researchops_agent.schemas.evaluation import AnswerEvalCase, AnswerEvalResult
from researchops_agent.schemas.llm import LLMProviderConfig


def _matched_substrings(expected: list[str], text: str) -> list[str]:
    lowered_text = text.lower()
    return [substring for substring in expected if substring.lower() in lowered_text]


def evaluate_llm_answers(
    cases: list[AnswerEvalCase],
    retriever_kind: str = "tfidf",
    provider: str = "fake",
    model: str | None = None,
) -> list[AnswerEvalResult]:
    results: list[AnswerEvalResult] = []
    for case in cases:
        evidence = build_evidence_from_document(
            case.document_path,
            case.query,
            retriever_kind=retriever_kind,
            top_k=5,
        )
        answer = answer_from_evidence_with_llm(
            evidence,
            LLMProviderConfig(provider=provider, model=model),
        )

        cited_text = " ".join(
            item.text for item in evidence.items if item.citation in answer.citations
        )
        supported_text = f"{answer.answer} {cited_text}".strip()
        matches = _matched_substrings(case.expected_answer_substrings, supported_text)

        if case.should_abstain:
            passed = answer.abstained
        else:
            passed = not answer.abstained and len(matches) == len(case.expected_answer_substrings)

        results.append(
            AnswerEvalResult(
                case_id=case.case_id,
                query=case.query,
                passed=passed,
                abstained=answer.abstained,
                matched_substrings=matches,
                citations=answer.citations,
                answer=answer.answer,
            )
        )

    return results
