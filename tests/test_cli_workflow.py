from typer.testing import CliRunner

from researchops_agent.cli import app

RUNNABLE_TEXT = (
    "We compare mean_baseline and Ridge for time_series_forecasting on synthetic_sine "
    "dataset using RMSE and MAE with chronological_split validation."
)


def _index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(RUNNABLE_TEXT, encoding="utf-8")
    index_dir = tmp_path / "index"
    CliRunner().invoke(app, ["index-corpus", str(docs), "--index-dir", str(index_dir)])
    return index_dir


def test_workflow_command_works_on_temp_corpus_index(tmp_path) -> None:
    index_dir = _index(tmp_path)
    out_dir = tmp_path / "workflows"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            str(index_dir),
            "time_series_forecasting synthetic_sine RMSE",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Workflow ID:" in result.output
    assert "Artifacts:" in result.output


def test_workflow_command_with_fake_llm_works(tmp_path) -> None:
    index_dir = _index(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            str(index_dir),
            "time_series_forecasting synthetic_sine Ridge RMSE",
            "--use-llm",
            "--llm-provider",
            "fake",
            "--out-dir",
            str(tmp_path / "workflows"),
        ],
    )

    assert result.exit_code == 0
    assert "Citations:" in result.output


def test_workflow_command_runs_bounded_experiment_when_runnable(tmp_path) -> None:
    index_dir = _index(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            str(index_dir),
            "time_series_forecasting synthetic_sine mean_baseline Ridge RMSE MAE",
            "--run-if-runnable",
            "--out-dir",
            str(tmp_path / "workflows"),
        ],
    )

    assert result.exit_code == 0
    assert "Run executed: True" in result.output
