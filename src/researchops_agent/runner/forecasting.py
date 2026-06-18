import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def make_lag_features(
    df: pd.DataFrame, target_col: str = "y", n_lags: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    features = {
        f"{target_col}_lag_{lag}": df[target_col].shift(lag)
        for lag in range(1, n_lags + 1)
    }
    supervised = pd.DataFrame(features)
    supervised[target_col] = df[target_col]
    supervised = supervised.dropna()

    X = supervised.drop(columns=[target_col])
    y = supervised[target_col]
    return X, y


def chronological_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def predict_mean_baseline(y_train: pd.Series, n_test: int) -> np.ndarray:
    return np.full(n_test, float(y_train.mean()))


def fit_predict_ridge(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    model = Ridge()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def compute_metric(name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_true) - np.asarray(y_pred)
    if name == "rmse":
        return float(np.sqrt(np.mean(errors**2)))
    if name == "mae":
        return float(np.mean(np.abs(errors)))
    raise ValueError(f"Unsupported metric: {name}")
