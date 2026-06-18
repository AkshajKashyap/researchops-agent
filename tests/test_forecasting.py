import numpy as np
import pandas as pd
import pytest

from researchops_agent.runner.datasets import make_synthetic_sine_series
from researchops_agent.runner.forecasting import (
    chronological_train_test_split,
    compute_metric,
    fit_predict_ridge,
    make_lag_features,
    predict_mean_baseline,
)


def test_synthetic_sine_generator_is_deterministic_with_same_seed() -> None:
    first = make_synthetic_sine_series(n_points=20, seed=7)
    second = make_synthetic_sine_series(n_points=20, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_lag_features_have_expected_columns_and_length() -> None:
    df = make_synthetic_sine_series(n_points=20, seed=7)

    X, y = make_lag_features(df, n_lags=3)

    assert list(X.columns) == ["y_lag_1", "y_lag_2", "y_lag_3"]
    assert len(X) == 17
    assert len(y) == 17


def test_chronological_split_preserves_order() -> None:
    df = make_synthetic_sine_series(n_points=20, seed=7)
    X, y = make_lag_features(df, n_lags=2)

    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=0.25)

    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.max() < y_test.index.min()


def test_mean_baseline_prediction_length_is_correct() -> None:
    y_train = pd.Series([1.0, 2.0, 3.0])

    predictions = predict_mean_baseline(y_train, n_test=4)

    assert len(predictions) == 4
    assert np.allclose(predictions, 2.0)


def test_ridge_prediction_length_is_correct() -> None:
    df = make_synthetic_sine_series(n_points=30, seed=7)
    X, y = make_lag_features(df)
    X_train, X_test, y_train, _ = chronological_train_test_split(X, y)

    predictions = fit_predict_ridge(X_train, y_train, X_test)

    assert len(predictions) == len(X_test)


def test_rmse_and_mae_compute_positive_values() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 5.0])

    assert compute_metric("rmse", y_true, y_pred) > 0
    assert compute_metric("mae", y_true, y_pred) > 0


def test_unsupported_metric_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_metric("accuracy", pd.Series([1.0]), np.array([1.0]))
