import json

from researchops_agent.schemas.workflow import WorkflowResult


def _list_or_none(items: list[str]) -> str:
    if not items:
        return "None"
    return "\n".join(f"- {item}" for item in items)


def format_workflow_markdown(result: WorkflowResult) -> str:
    lines: list[str] = [
        "# ResearchOps Workflow Report",
        "",
        f"- Workflow ID: `{result.workflow_id}`",
        f"- Query: {result.query}",
        f"- Corpus index: `{result.corpus_index_dir}`",
        "",
        "## Options",
        "",
        "| Option | Value |",
        "| --- | --- |",
    ]

    for key, value in result.options.model_dump().items():
        lines.append(f"| {key} | `{value}` |")

    lines.extend(
        [
            "",
            "## Steps",
            "",
            "| Step | Status | Message |",
            "| --- | --- | --- |",
        ]
    )
    for step in result.steps:
        lines.append(f"| {step.name} | {step.status} | {step.message or ''} |")

    lines.extend(
        [
            "",
            "## Answer",
            "",
            result.answer,
            "",
            f"Abstained: `{result.abstained}`",
            "",
            "## Citations",
            "",
            _list_or_none(result.citations),
            "",
            "## Claims",
            "",
        ]
    )

    if result.claims:
        lines.extend(["| Claim | Dataset | Models | Metrics |", "| --- | --- | --- | --- |"])
        for claim in result.claims:
            lines.append(
                "| "
                f"{claim.main_claim} | "
                f"{claim.dataset or ''} | "
                f"{', '.join(claim.models)} | "
                f"{', '.join(claim.metrics)} |"
            )
    else:
        lines.append("None")

    lines.extend(["", "## Suggested Config", ""])
    if result.suggested_config is None:
        lines.append("None")
    else:
        lines.extend(
            [
                "```json",
                json.dumps(result.suggested_config.model_dump(), indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Config Validation",
            "",
            f"Runnable: `{result.config_validation.runnable}`",
            "",
            _list_or_none(result.config_validation.errors),
            "",
            "## Bounded Run",
            "",
        ]
    )

    if result.run_result is None:
        lines.append("No bounded run was executed.")
    else:
        lines.extend(
            [
                f"- Run ID: `{result.run_result.run_id}`",
                f"- Status: {result.run_result.status}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for metric in result.run_result.metrics:
            lines.append(f"| {metric.name} | {metric.value:.6f} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    if result.artifacts:
        lines.extend(["| Name | Type | Path |", "| --- | --- | --- |"])
        for artifact in result.artifacts:
            lines.append(f"| {artifact.name} | {artifact.artifact_type} | `{artifact.path}` |")
    else:
        lines.append("None")

    lines.extend(
        [
            "",
            "## Honesty Notes",
            "",
            _list_or_none(result.honesty_notes),
            "",
        ]
    )
    return "\n".join(lines)
