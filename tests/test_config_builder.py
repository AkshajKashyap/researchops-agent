from researchops_agent.runner.config_builder import suggest_experiment_config
from researchops_agent.schemas.experiment import ExperimentClaim


def test_returns_none_for_empty_claims() -> None:
    assert suggest_experiment_config([]) is None


def test_creates_config_from_claim() -> None:
    claim = ExperimentClaim(
        task="classification",
        dataset="MIMIC",
        models=["Ridge", "LSTM"],
        metrics=["accuracy", "F1"],
        main_claim="We compare Ridge and LSTM on the MIMIC dataset with cross-validation.",
    )

    config = suggest_experiment_config([claim])

    assert config is not None
    assert config.task == "classification"
    assert config.dataset == "MIMIC"
    assert config.validation_strategy == "cross_validation"


def test_includes_missing_information_notes() -> None:
    claim = ExperimentClaim(main_claim="Results show accuracy improves.")

    config = suggest_experiment_config([claim])

    assert config is not None
    assert "Dataset was not clearly specified in evidence." in config.notes
    assert "Validation strategy was not clearly specified in evidence." in config.notes
    assert "Suggested config is heuristic and should be reviewed before running." in config.notes


def test_preserves_model_and_metric_lists() -> None:
    claim = ExperimentClaim(
        main_claim="We evaluate Ridge with RMSE.",
        models=["Ridge"],
        metrics=["RMSE"],
    )

    config = suggest_experiment_config([claim])

    assert config is not None
    assert config.models == ["Ridge"]
    assert config.metrics == ["RMSE"]
