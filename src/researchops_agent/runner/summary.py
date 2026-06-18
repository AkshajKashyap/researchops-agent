from researchops_agent.schemas.run import ExperimentRunResult


def format_run_summary(result: ExperimentRunResult) -> str:
    lines = [
        "# Experiment Run Summary",
        "",
        f"Run ID: `{result.run_id}`",
        f"Status: `{result.status}`",
        f"Task: `{result.task}`",
        f"Dataset: `{result.dataset or 'not_specified'}`",
        f"Models: {', '.join(result.models) if result.models else 'not_specified'}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]

    for metric in result.metrics:
        lines.append(f"| {metric.name} | {metric.value:.6f} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Name | Type | Path |",
            "| --- | --- | --- |",
        ]
    )

    for artifact in result.artifacts:
        lines.append(f"| {artifact.name} | {artifact.artifact_type} | {artifact.path} |")

    lines.extend(
        [
            "",
            "## Honesty Note",
            "",
            "This is a bounded local runner for a small supported experiment family, "
            "not a general paper reproduction engine.",
            "",
        ]
    )
    return "\n".join(lines)
