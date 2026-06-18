from researchops_agent.schemas.experiment import ExperimentConfig
from researchops_agent.workflows.config_check import check_config_runnable


def test_none_config_is_not_runnable() -> None:
    result = check_config_runnable(None)

    assert result.runnable is False
    assert result.errors


def test_valid_runnable_config_passes() -> None:
    config = ExperimentConfig(
        task="time_series_forecasting",
        dataset="synthetic_sine",
        models=["mean_baseline", "ridge"],
        metrics=["rmse", "mae"],
        validation_strategy="chronological_split",
    )

    result = check_config_runnable(config)

    assert result.runnable is True
    assert result.errors == []


def test_unsupported_config_returns_not_runnable_without_raising() -> None:
    config = ExperimentConfig(
        task="classification",
        dataset="synthetic_sine",
        models=["ridge"],
        metrics=["rmse"],
        validation_strategy="chronological_split",
    )

    result = check_config_runnable(config)

    assert result.runnable is False
    assert "Unsupported task" in result.errors[0]
