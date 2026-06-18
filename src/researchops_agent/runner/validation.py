from researchops_agent.schemas.experiment import ExperimentConfig

SUPPORTED_TASKS = {"time_series_forecasting"}
SUPPORTED_DATASETS = {"synthetic_sine"}
SUPPORTED_MODELS = {"mean_baseline", "ridge"}
SUPPORTED_METRICS = {"rmse", "mae"}
SUPPORTED_VALIDATION_STRATEGIES = {"chronological_split"}


def validate_runnable_config(config: ExperimentConfig) -> None:
    if config.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {config.task}")
    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    if not config.models:
        raise ValueError("At least one model must be provided")
    if not config.metrics:
        raise ValueError("At least one metric must be provided")
    if config.validation_strategy not in SUPPORTED_VALIDATION_STRATEGIES:
        raise ValueError(f"Unsupported validation strategy: {config.validation_strategy}")

    for model in config.models:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
    for metric in config.metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")
