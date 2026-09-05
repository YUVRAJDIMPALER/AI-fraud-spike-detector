from __future__ import annotations

import pandas as pd


def time_split(bucket: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "step" not in bucket.columns:
        raise ValueError("Bucket data must include a 'step' column for time-based splitting.")

    sorted_steps = sorted(bucket["step"].unique())
    cutoff_index = max(1, int(len(sorted_steps) * (1 - test_fraction)))
    cutoff_step = sorted_steps[cutoff_index - 1]

    train = bucket[bucket["step"] <= cutoff_step].copy()
    test = bucket[bucket["step"] > cutoff_step].copy()
    return train, test
