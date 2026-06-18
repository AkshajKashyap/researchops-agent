from researchops_agent.schemas.evaluation import (
    AnswerEvalCase,
    EvaluationReport,
    EvaluationSummary,
    RetrievalEvalCase,
)


def test_evaluation_schemas_validate_expected_fields() -> None:
    retrieval_case = RetrievalEvalCase(
        case_id="retrieval-1",
        document_path="doc.md",
        query="What models are used?",
        expected_substrings=["Ridge"],
    )
    answer_case = AnswerEvalCase(
        case_id="answer-1",
        document_path="doc.md",
        query="What metrics are used?",
        expected_answer_substrings=["RMSE"],
    )
    summary = EvaluationSummary(
        retrieval_cases=1,
        retrieval_hits=1,
        retrieval_hit_rate=1.0,
        answer_cases=1,
        answer_passes=1,
        answer_pass_rate=1.0,
    )

    report = EvaluationReport(summary=summary)

    assert retrieval_case.expected_citations == []
    assert answer_case.should_abstain is False
    assert report.retrieval_results == []
    assert report.answer_results == []
