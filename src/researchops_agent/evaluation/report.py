from researchops_agent.schemas.evaluation import (
    AnswerEvalResult,
    EvaluationReport,
    EvaluationSummary,
    RetrievalEvalResult,
)


def build_evaluation_report(
    retrieval_results: list[RetrievalEvalResult],
    answer_results: list[AnswerEvalResult],
) -> EvaluationReport:
    retrieval_hits = sum(result.hit for result in retrieval_results)
    answer_passes = sum(result.passed for result in answer_results)
    retrieval_cases = len(retrieval_results)
    answer_cases = len(answer_results)

    summary = EvaluationSummary(
        retrieval_cases=retrieval_cases,
        retrieval_hits=retrieval_hits,
        retrieval_hit_rate=retrieval_hits / retrieval_cases if retrieval_cases else 0.0,
        answer_cases=answer_cases,
        answer_passes=answer_passes,
        answer_pass_rate=answer_passes / answer_cases if answer_cases else 0.0,
    )

    return EvaluationReport(
        summary=summary,
        retrieval_results=retrieval_results,
        answer_results=answer_results,
    )


def _cell(value: object) -> str:
    return " ".join(str(value).replace("|", "\\|").split())


def format_evaluation_markdown(report: EvaluationReport) -> str:
    lines = [
        "# ResearchOps Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Retrieval cases | {report.summary.retrieval_cases} |",
        f"| Retrieval hits | {report.summary.retrieval_hits} |",
        f"| Retrieval hit rate | {report.summary.retrieval_hit_rate:.2f} |",
        f"| Answer cases | {report.summary.answer_cases} |",
        f"| Answer passes | {report.summary.answer_passes} |",
        f"| Answer pass rate | {report.summary.answer_pass_rate:.2f} |",
        "",
        "## Retrieval Results",
        "",
        "| Case | Hit | Matches | Top citations |",
        "| --- | --- | --- | --- |",
    ]

    for result in report.retrieval_results:
        lines.append(
            "| "
            f"{_cell(result.case_id)} | "
            f"{result.hit} | "
            f"{_cell(', '.join(result.matched_substrings))} | "
            f"{_cell(', '.join(result.top_citations))} |"
        )

    lines.extend(
        [
            "",
            "## Answer Results",
            "",
            "| Case | Passed | Abstained | Matches | Citations |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for result in report.answer_results:
        lines.append(
            "| "
            f"{_cell(result.case_id)} | "
            f"{result.passed} | "
            f"{result.abstained} | "
            f"{_cell(', '.join(result.matched_substrings))} | "
            f"{_cell(', '.join(result.citations))} |"
        )

    lines.extend(
        [
            "",
            "## Honesty Note",
            "",
            "This is a deterministic local eval harness. It measures basic evidence retrieval "
            "and extractive answer grounding, not full natural-language answer quality.",
            "",
        ]
    )

    return "\n".join(lines)
