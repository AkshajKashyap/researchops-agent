from typer.testing import CliRunner

from researchops_agent.cli import app
from researchops_agent.utils.yaml_io import read_yaml


def test_run_config_command_works_with_temp_out_dir(tmp_path) -> None:
    out_dir = tmp_path / "runs"

    result = CliRunner().invoke(
        app,
        [
            "run-config",
            "configs/time_series_demo.yaml",
            "--out-dir",
            str(out_dir),
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0
    assert "Run ID:" in result.output
    assert "Status: success" in result.output
    assert "ridge_rmse" in result.output


def test_suggest_config_command_writes_yaml_from_temp_markdown(tmp_path) -> None:
    document = tmp_path / "experiment.md"
    output = tmp_path / "configs" / "suggested.yaml"
    document.write_text(
        "We compare mean_baseline and Ridge on the synthetic_sine dataset for "
        "time_series_forecasting using RMSE and MAE with chronological_split validation.",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "suggest-config",
            str(document),
            "synthetic_sine time_series_forecasting RMSE MAE",
            "--out",
            str(output),
            "--retriever",
            "tfidf",
        ],
    )

    assert result.exit_code == 0
    assert "Suggested config:" in result.output
    assert output.exists()
    data = read_yaml(str(output))
    assert data["task"] == "time_series_forecasting"
    assert data["dataset"] == "synthetic_sine"
    assert data["models"] == ["mean_baseline", "ridge"]
    assert data["metrics"] == ["rmse", "mae"]
    assert data["validation_strategy"] == "chronological_split"
