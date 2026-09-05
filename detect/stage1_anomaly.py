from __future__ import annotations

import numpy as np
import pandas as pd


def compute_baseline(bucket: pd.DataFrame, metric: str = "txn_count") -> tuple[float, float]:
    mean_val = bucket[metric].mean()
    std_val = bucket[metric].std(ddof=0)
    if std_val == 0:
        std_val = 1e-6
    return mean_val, std_val


def ewma_zscore(series: pd.Series, alpha: float = 0.2) -> pd.Series:
    ewma = series.ewm(alpha=alpha, adjust=False).mean()
    sigma = series.ewm(alpha=alpha, adjust=False).std()
    sigma = sigma.replace(0, np.nan)
    return ((series - ewma) / sigma).fillna(0)


def cusum_signal(series: pd.Series, threshold: float = 3.0, drift: float = 0.0) -> pd.Series:
    s = series.astype(float)
    pos_cusum = np.zeros(len(s))
    neg_cusum = np.zeros(len(s))
    for i in range(1, len(s)):
        pos_cusum[i] = max(0, pos_cusum[i - 1] + (s.iloc[i] - s.iloc[i - 1] - drift))
        neg_cusum[i] = max(0, neg_cusum[i - 1] - (s.iloc[i] - s.iloc[i - 1] + drift))
    return pd.Series(np.maximum(pos_cusum, neg_cusum), index=s.index)


def stage1_flag(bucket: pd.DataFrame, metric: str = "txn_count") -> pd.DataFrame:
    base = bucket.copy()
    mu, sigma = compute_baseline(base, metric)
    base["zscore"] = (base[metric] - mu) / sigma
    base["ewma_z"] = ewma_zscore(base[metric])
    base["cusum"] = cusum_signal(base["zscore"], threshold=3.0, drift=0.5)
    base["anomaly_flag"] = (
        (base["zscore"].abs() > 2.5) |
        (base["ewma_z"].abs() > 2.5) |
        (base["cusum"] > 3.0)
    ).astype(int)
    return base
