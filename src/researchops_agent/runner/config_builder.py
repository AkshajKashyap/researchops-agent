import re

from researchops_agent.schemas.experiment import ExperimentClaim, ExperimentConfig


def _validation_strategy(text: str) -> str:
    lowered = text.lower()
    if "cross-validation" in lowered or "cross validation" in lowered:
        return "cross_validation"
    if "holdout" in lowered:
        return "holdout"
    if "validation" in lowered:
        match = re.search(r"([^.]*validation[^.]*)", text, flags=re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())
    return "not_specified"


def suggest_experiment_config(claims: list[ExperimentClaim]) -> ExperimentConfig | None:
    if not claims:
        return None

    claim = claims[0]
    validation_strategy = _validation_strategy(claim.main_claim)
    notes = ["Suggested config is heuristic and should be reviewed before running."]

    if not claim.dataset:
        notes.append("Dataset was not clearly specified in evidence.")
    if not claim.models:
        notes.append("Models were not clearly specified in evidence.")
    if not claim.metrics:
        notes.append("Metrics were not clearly specified in evidence.")
    if validation_strategy == "not_specified":
        notes.append("Validation strategy was not clearly specified in evidence.")

    return ExperimentConfig(
        task=claim.task or "not_specified",
        dataset=claim.dataset,
        models=claim.models,
        metrics=claim.metrics,
        validation_strategy=validation_strategy,
        notes=notes,
    )
