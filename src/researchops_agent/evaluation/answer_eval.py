from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.retrieval import factory as retriever_factory
from researchops_agent.schemas.evaluation import AnswerEvalCase, AnswerEvalResult


def _matched_substrings(expected: list[str], text: str) -> list[str]:
    lowered_text = text.lower()
    return [substring for substring in expected if substring.lower() in lowered_text]


def evaluate_answers(
    cases: list[AnswerEvalCase], retriever_kind: str = "tfidf"
) -> list[AnswerEvalResult]:
    results: list[AnswerEvalResult] = []
    for case in cases:
        pages = load_document(case.document_path)
        chunks = chunk_pages(pages)
        retriever = retriever_factory.build_retriever(retriever_kind)
        retriever.fit(chunks)
        retrieval = retriever.search(case.query, top_k=5)
        evidence = build_evidence_pack(retrieval, min_score=0.05)
        answer = answer_from_evidence(evidence)

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
