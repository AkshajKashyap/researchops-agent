from pathlib import Path

import pytest

from researchops_agent.runner.experiment_runner import run_experiment_config
from researchops_agent.runner.summary import format_run_summary
from researchops_agent.schemas.experiment import ExperimentConfig


def _config(**overrides) -> ExperimentConfig:
    data = {
        "task": "time_series_forecasting",
        "dataset": "synthetic_sine",
        "models": ["mean_baseline", "ridge"],
        "metrics": ["rmse", "mae"],
        "validation_strategy": "chronological_split",
        "notes": ["test config"],
    }
    data.update(overrides)
    return ExperimentConfig(**data)


def test_valid_config_runs_successfully(tmp_path) -> None:
    result = run_experiment_config(_config(), out_dir=str(tmp_path), seed=123)

    assert result.status == "success"
    assert result.task == "time_series_forecasting"
    assert result.dataset == "synthetic_sine"


def test_metrics_include_model_metric_names(tmp_path) -> None:
    result = run_experiment_config(_config(), out_dir=str(tmp_path), seed=123)

    metric_names = {metric.name for metric in result.metrics}

    assert metric_names == {
        "mean_baseline_rmse",
        "mean_baseline_mae",
        "ridge_rmse",
        "ridge_mae",
    }


def test_artifacts_are_written(tmp_path) -> None:
    result = run_experiment_config(_config(), out_dir=str(tmp_path), seed=123)

    for artifact in result.artifacts:
        assert artifact.path
        assert artifact.artifact_type
        assert artifact.name
        assert artifact.path.startswith(str(tmp_path))
        assert Path(artifact.path).exists()


def test_summary_markdown_includes_honesty_note(tmp_path) -> None:
    result = run_experiment_config(_config(), out_dir=str(tmp_path), seed=123)

    summary = format_run_summary(result)

    assert "bounded local runner" in summary
    assert "not a general paper reproduction engine" in summary


def test_invalid_config_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        run_experiment_config(_config(models=["lstm"]), out_dir=str(tmp_path))
