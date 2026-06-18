from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.evaluation.report import build_evaluation_report
from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.schemas.evaluation import (
    AnswerEvalCase,
    EvaluationReport,
    RetrievalEvalCase,
)


def compare_retrievers(
    retrieval_cases: list[RetrievalEvalCase],
    answer_cases: list[AnswerEvalCase],
    retriever_kinds: list[str],
    top_k: int = 3,
) -> dict[str, EvaluationReport]:
    if not retriever_kinds:
        raise ValueError("retriever_kinds must not be empty")

    reports: dict[str, EvaluationReport] = {}
    for retriever_kind in retriever_kinds:
        retrieval_results = evaluate_retrieval(
            retrieval_cases, top_k=top_k, retriever_kind=retriever_kind
        )
        answer_results = evaluate_answers(answer_cases, retriever_kind=retriever_kind)
        reports[retriever_kind] = build_evaluation_report(retrieval_results, answer_results)

    return reports
