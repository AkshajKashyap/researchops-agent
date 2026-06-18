from researchops_agent.schemas.workflow import (
    ConfigValidationResult,
    WorkflowOptions,
    WorkflowResult,
    WorkflowStep,
)
from researchops_agent.workflows.reporting import format_workflow_markdown


def test_workflow_markdown_contains_key_sections() -> None:
    result = WorkflowResult(
        workflow_id="workflow_123",
        query="What happened?",
        corpus_index_dir="data/indexes/demo",
        options=WorkflowOptions(),
        steps=[WorkflowStep(name="corpus_search", status="success")],
        answer="Ridge was compared with a baseline.",
        citations=["note.md#chunk=0"],
        abstained=False,
        config_validation=ConfigValidationResult(runnable=False, errors=["Unsupported task"]),
        honesty_notes=["Local evidence only."],
    )

    markdown = format_workflow_markdown(result)

    assert "workflow_123" in markdown
    assert "Ridge was compared" in markdown
    assert "note.md#chunk=0" in markdown
    assert "Local evidence only." in markdown
