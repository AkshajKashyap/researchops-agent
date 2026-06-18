import json

from researchops_agent.corpus.search import build_index_for_corpus
from researchops_agent.schemas.workflow import WorkflowOptions
from researchops_agent.workflows.orchestrator import run_research_workflow

RUNNABLE_TEXT = (
    "We compare mean_baseline and Ridge for time_series_forecasting on synthetic_sine "
    "dataset using RMSE and MAE with chronological_split validation. Results show "
    "Ridge improves RMSE over the baseline."
)
RUNNABLE_QUERY = "time_series_forecasting synthetic_sine mean_baseline Ridge RMSE MAE"


def _index(tmp_path, text: str) -> str:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(text, encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index_for_corpus(str(docs), str(index_dir))
    return str(index_dir)


def test_workflow_id_is_deterministic(tmp_path) -> None:
    index_dir = _index(tmp_path, RUNNABLE_TEXT)
    out_dir = tmp_path / "workflows"

    first = run_research_workflow(index_dir, RUNNABLE_QUERY, out_dir=str(out_dir))
    second = run_research_workflow(index_dir, RUNNABLE_QUERY, out_dir=str(out_dir))

    assert first.workflow_id == second.workflow_id


def test_workflow_writes_artifacts_and_suggested_config(tmp_path) -> None:
    index_dir = _index(tmp_path, RUNNABLE_TEXT)
    result = run_research_workflow(index_dir, RUNNABLE_QUERY, out_dir=str(tmp_path / "workflows"))
    workflow_dir = tmp_path / "workflows" / result.workflow_id

    assert (workflow_dir / "workflow_result.json").exists()
    assert (workflow_dir / "workflow_summary.md").exists()
    assert (workflow_dir / "suggested_config.yaml").exists()
    assert result.suggested_config is not None
    assert result.honesty_notes

    payload = json.loads((workflow_dir / "workflow_result.json").read_text(encoding="utf-8"))
    assert payload["workflow_id"] == result.workflow_id


def test_workflow_skips_bounded_run_when_config_is_not_runnable(tmp_path) -> None:
    text = (
        "We compare SVM and WL graph kernels on MUTAG dataset using accuracy with "
        "repeated stratified validation. Results show WL is stronger."
    )
    index_dir = _index(tmp_path, text)

    result = run_research_workflow(
        index_dir,
        "SVM WL MUTAG accuracy validation",
        options=WorkflowOptions(run_if_runnable=True),
        out_dir=str(tmp_path / "workflows"),
    )

    assert result.config_validation.runnable is False
    assert result.run_result is None
    assert any(step.name == "bounded_run" and step.status == "skipped" for step in result.steps)


def test_workflow_executes_bounded_run_when_config_is_runnable(tmp_path) -> None:
    index_dir = _index(tmp_path, RUNNABLE_TEXT)

    result = run_research_workflow(
        index_dir,
        RUNNABLE_QUERY,
        options=WorkflowOptions(run_if_runnable=True),
        out_dir=str(tmp_path / "workflows"),
    )

    assert result.config_validation.runnable is True
    assert result.run_result is not None
    assert result.run_result.status == "success"
    assert any(metric.name == "ridge_rmse" for metric in result.run_result.metrics)


def test_llm_workflow_works_with_fake_provider(tmp_path) -> None:
    index_dir = _index(tmp_path, RUNNABLE_TEXT)

    result = run_research_workflow(
        index_dir,
        RUNNABLE_QUERY,
        options=WorkflowOptions(use_llm=True, llm_provider="fake"),
        out_dir=str(tmp_path / "workflows"),
    )

    assert result.abstained is False
    assert result.citations
