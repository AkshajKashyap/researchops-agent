import hashlib
import json
from pathlib import Path

import pandas as pd

from researchops_agent.runner.datasets import make_synthetic_sine_series
from researchops_agent.runner.forecasting import (
    chronological_train_test_split,
    compute_metric,
    fit_predict_ridge,
    make_lag_features,
    predict_mean_baseline,
)
from researchops_agent.runner.summary import format_run_summary
from researchops_agent.runner.validation import validate_runnable_config
from researchops_agent.schemas.experiment import ExperimentConfig
from researchops_agent.schemas.run import ExperimentRunResult, RunArtifact, RunMetric
from researchops_agent.utils.json_io import write_json
from researchops_agent.utils.text_io import write_text
from researchops_agent.utils.yaml_io import write_yaml


def _run_id(config: ExperimentConfig, seed: int) -> str:
    payload = {"config": config.model_dump(), "seed": seed}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{config.task}_{config.dataset}_{digest}"


def _predict_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> pd.Series:
    if model_name == "mean_baseline":
        predictions = predict_mean_baseline(y_train, len(X_test))
    elif model_name == "ridge":
        predictions = fit_predict_ridge(X_train, y_train, X_test)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return pd.Series(predictions, index=X_test.index, name=model_name)


def run_experiment_config(
    config: ExperimentConfig, out_dir: str = "reports/runs", seed: int = 42
) -> ExperimentRunResult:
    validate_runnable_config(config)

    run_id = _run_id(config, seed)
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.dataset == "synthetic_sine":
        df = make_synthetic_sine_series(seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    X, y = make_lag_features(df)
    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y)

    metrics: list[RunMetric] = []
    prediction_frames: list[pd.Series] = []
    for model_name in config.models:
        y_pred = _predict_model(model_name, X_train, X_test, y_train)
        prediction_frames.append(y_pred)
        for metric_name in config.metrics:
            metrics.append(
                RunMetric(
                    name=f"{model_name}_{metric_name}",
                    value=compute_metric(metric_name, y_test, y_pred.to_numpy()),
                )
            )

    predictions = pd.concat(prediction_frames, axis=1)
    predictions.insert(0, "y_true", y_test)

    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    summary_path = run_dir / "summary.md"

    artifacts = [
        RunArtifact(name="config", path=str(config_path), artifact_type="yaml"),
        RunArtifact(name="metrics", path=str(metrics_path), artifact_type="json"),
        RunArtifact(name="predictions", path=str(predictions_path), artifact_type="csv"),
        RunArtifact(name="summary", path=str(summary_path), artifact_type="markdown"),
    ]

    result = ExperimentRunResult(
        run_id=run_id,
        task=config.task,
        dataset=config.dataset,
        models=config.models,
        metrics=metrics,
        artifacts=artifacts,
        status="success",
        message="Experiment completed successfully.",
    )

    write_yaml(str(config_path), config.model_dump())
    write_json(str(metrics_path), {"metrics": [metric.model_dump() for metric in metrics]})
    predictions.to_csv(predictions_path, index=False)
    write_text(str(summary_path), format_run_summary(result))

    return result
