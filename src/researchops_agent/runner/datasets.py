import numpy as np
import pandas as pd


def make_synthetic_sine_series(n_points: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    seasonal = np.sin(2 * np.pi * t / 24)
    trend = 0.002 * t
    noise = rng.normal(loc=0.0, scale=0.08, size=n_points)
    y = seasonal + trend + noise
    return pd.DataFrame({"t": t, "y": y})
