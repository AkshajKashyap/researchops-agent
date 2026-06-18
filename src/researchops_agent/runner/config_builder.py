import re

from researchops_agent.schemas.experiment import ExperimentClaim, ExperimentConfig


def _validation_strategy(text: str) -> str:
    lowered = text.lower()
    if "chronological_split" in lowered or "chronological split" in lowered:
        return "chronological_split"
    if "cross-validation" in lowered or "cross validation" in lowered:
        return "cross_validation"
    if "holdout" in lowered:
        return "holdout"
    if "validation" in lowered:
        match = re.search(r"([^.]*validation[^.]*)", text, flags=re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())
    return "not_specified"


def _runnable_model_name(model: str) -> str:
    lowered = model.lower()
    if lowered == "ridge":
        return "ridge"
    return model


def _runnable_metric_name(metric: str) -> str:
    lowered = metric.lower()
    if lowered in {"rmse", "mae"}:
        return lowered
    return metric


def _claim_priority(claim: ExperimentClaim) -> tuple[int, int, int, int, int]:
    validation_strategy = _validation_strategy(claim.main_claim)
    return (
        int(claim.task is not None),
        int(claim.dataset is not None),
        int(validation_strategy != "not_specified"),
        len(claim.models),
        len(claim.metrics),
    )


def _first_present(claims: list[ExperimentClaim], field: str) -> str | None:
    for claim in claims:
        value = getattr(claim, field)
        if value:
            return value
    return None


def _first_non_empty_list(claims: list[ExperimentClaim], field: str) -> list[str]:
    for claim in claims:
        values = getattr(claim, field)
        if values:
            return values
    return []


def _first_validation_strategy(claims: list[ExperimentClaim]) -> str:
    for claim in claims:
        validation_strategy = _validation_strategy(claim.main_claim)
        if validation_strategy != "not_specified":
            return validation_strategy
    return "not_specified"


def suggest_experiment_config(claims: list[ExperimentClaim]) -> ExperimentConfig | None:
    if not claims:
        return None

    claim = max(claims, key=_claim_priority)
    task = claim.task or _first_present(claims, "task")
    dataset = claim.dataset or _first_present(claims, "dataset")
    models = claim.models or _first_non_empty_list(claims, "models")
    metrics = claim.metrics or _first_non_empty_list(claims, "metrics")
    validation_strategy = _validation_strategy(claim.main_claim)
    if validation_strategy == "not_specified":
        validation_strategy = _first_validation_strategy(claims)
    has_runnable_hint = (
        task == "time_series_forecasting"
        or dataset == "synthetic_sine"
        or validation_strategy == "chronological_split"
    )
    notes = ["Suggested config is heuristic and should be reviewed before running."]

    if not dataset:
        notes.append("Dataset was not clearly specified in evidence.")
    if not models:
        notes.append("Models were not clearly specified in evidence.")
    if not metrics:
        notes.append("Metrics were not clearly specified in evidence.")
    if validation_strategy == "not_specified":
        notes.append("Validation strategy was not clearly specified in evidence.")

    return ExperimentConfig(
        task=task or "not_specified",
        dataset=dataset,
        models=[
            _runnable_model_name(model) if has_runnable_hint else model
            for model in models
        ],
        metrics=[
            _runnable_metric_name(metric) if has_runnable_hint else metric
            for metric in metrics
        ],
        validation_strategy=validation_strategy,
        notes=notes,
    )
