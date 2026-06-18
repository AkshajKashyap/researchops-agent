from researchops_agent.runner.validation import validate_runnable_config
from researchops_agent.schemas.experiment import ExperimentConfig
from researchops_agent.schemas.workflow import ConfigValidationResult


def check_config_runnable(config: ExperimentConfig | None) -> ConfigValidationResult:
    if config is None:
        return ConfigValidationResult(
            runnable=False,
            errors=["No experiment config was suggested from retrieved evidence."],
        )

    try:
        validate_runnable_config(config)
    except ValueError as exc:
        return ConfigValidationResult(runnable=False, errors=[str(exc)])

    return ConfigValidationResult(runnable=True)
