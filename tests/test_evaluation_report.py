from researchops_agent.evaluation.report import (
    build_evaluation_report,
    format_evaluation_markdown,
)
from researchops_agent.schemas.evaluation import AnswerEvalResult, RetrievalEvalResult


def test_evaluation_report_summary_counts_are_correct() -> None:
    report = build_evaluation_report(
        [
            RetrievalEvalResult(case_id="r1", query="q", top_k=1, hit=True),
            RetrievalEvalResult(case_id="r2", query="q", top_k=1, hit=False),
        ],
        [
            AnswerEvalResult(case_id="a1", query="q", passed=True, abstained=False, answer="a"),
            AnswerEvalResult(case_id="a2", query="q", passed=False, abstained=True, answer="a"),
        ],
    )

    assert report.summary.retrieval_cases == 2
    assert report.summary.retrieval_hits == 1
    assert report.summary.answer_cases == 2
    assert report.summary.answer_passes == 1


def test_evaluation_report_rates_are_calculated_correctly() -> None:
    report = build_evaluation_report(
        [
            RetrievalEvalResult(case_id="r1", query="q", top_k=1, hit=True),
            RetrievalEvalResult(case_id="r2", query="q", top_k=1, hit=False),
        ],
        [AnswerEvalResult(case_id="a1", query="q", passed=True, abstained=False, answer="a")],
    )

    assert report.summary.retrieval_hit_rate == 0.5
    assert report.summary.answer_pass_rate == 1.0


def test_markdown_report_includes_summary_and_tables() -> None:
    report = build_evaluation_report(
        [RetrievalEvalResult(case_id="r1", query="q", top_k=1, hit=True)],
        [AnswerEvalResult(case_id="a1", query="q", passed=True, abstained=False, answer="a")],
    )

    markdown = format_evaluation_markdown(report)

    assert "# ResearchOps Evaluation Report" in markdown
    assert "## Summary" in markdown
    assert "| Retrieval Results" not in markdown
    assert "## Retrieval Results" in markdown
    assert "## Answer Results" in markdown
    assert "deterministic local eval harness" in markdown
