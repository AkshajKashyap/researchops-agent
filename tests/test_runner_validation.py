import pytest

from researchops_agent.runner.validation import validate_runnable_config
from researchops_agent.schemas.experiment import ExperimentConfig


def _config(**overrides) -> ExperimentConfig:
    data = {
        "task": "time_series_forecasting",
        "dataset": "synthetic_sine",
        "models": ["mean_baseline", "ridge"],
        "metrics": ["rmse", "mae"],
        "validation_strategy": "chronological_split",
    }
    data.update(overrides)
    return ExperimentConfig(**data)


def test_valid_config_passes() -> None:
    validate_runnable_config(_config())


def test_unsupported_task_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        validate_runnable_config(_config(task="classification"))


def test_unsupported_dataset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        validate_runnable_config(_config(dataset="mimic"))


def test_unsupported_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        validate_runnable_config(_config(models=["lstm"]))


def test_unsupported_metric_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        validate_runnable_config(_config(metrics=["accuracy"]))


def test_unsupported_validation_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported validation strategy"):
        validate_runnable_config(_config(validation_strategy="random_split"))


def test_empty_models_raises_value_error() -> None:
    with pytest.raises(ValueError, match="At least one model"):
        validate_runnable_config(_config(models=[]))


def test_empty_metrics_raises_value_error() -> None:
    with pytest.raises(ValueError, match="At least one metric"):
        validate_runnable_config(_config(metrics=[]))
