from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from simulate.config import CONFIG
from simulate.inject_spikes import apply_injections


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "buckets_labeled.csv"


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        candidates = [
            Path("data/raw") / "PS_20174392719_1491204439457_log.csv",
            Path("data/raw") / "paysim.csv",
            Path("data/raw") / "PS_20174392719_1491204439457_log.csv.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "No PaySim CSV was found in data/raw. Place the original file there before running the pipeline."
            )

    df = pd.read_csv(path)
    if "step" not in df.columns:
        raise ValueError("The PaySim file must include a 'step' column for bucketing.")
    return df


def create_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    bucket = (
        df.groupby("step", as_index=False)
        .agg(
            txn_count=("amount", "count"),
            unique_senders=("nameOrig", "nunique"),
            avg_amount=("amount", "mean"),
            total_amount=("amount", "sum"),
            fraud_count=("isFraud", "sum"),
            payment_count=("type", lambda s: (s == "PAYMENT").sum()),
            transfer_count=("type", lambda s: (s == "TRANSFER").sum()),
        )
    )

    bucket["unique_ratio"] = bucket["unique_senders"] / bucket["txn_count"]
    bucket["amount_ratio"] = bucket["total_amount"] / (bucket["txn_count"].replace(0, 1))
    bucket["fraud_rate"] = bucket["fraud_count"] / bucket["txn_count"].replace(0, 1)

    # NEW: how "concentrated" is this bucket's traffic on one IP or BIN?
    # A high share here means most transactions came from the SAME
    # source - a strong signal of a single bad actor or bot farm.
    if "ip_address" in df.columns:
        ip_concentration = (
            df.groupby("step")["ip_address"]
            .apply(lambda s: s.value_counts(normalize=True).max() if s.notna().any() else 0.0)
            .reset_index(name="top_ip_share")
        )
        bin_concentration = (
            df.groupby("step")["bin_number"]
            .apply(lambda s: s.value_counts(normalize=True).max() if s.notna().any() else 0.0)
            .reset_index(name="top_bin_share")
        )
        bucket = bucket.merge(ip_concentration, on="step", how="left")
        bucket = bucket.merge(bin_concentration, on="step", how="left")
        bucket["top_ip_share"] = bucket["top_ip_share"].fillna(0.0)
        bucket["top_bin_share"] = bucket["top_bin_share"].fillna(0.0)

    if "fraud_type" in df.columns:
        fraud_type_per_step = (
        df[df["fraud_type"].notna()]
        .groupby("step")["fraud_type"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .reset_index()
    )
    bucket = bucket.merge(fraud_type_per_step, on="step", how="left")    

    return bucket


def label_bucket(step: int, fraud_steps: set[int], demand_steps: set[int]) -> str:
    if step in fraud_steps:
        return "fraud_attack"
    if step in demand_steps:
        return "genuine_demand"
    return "normal"


def build_labeled_buckets(df: pd.DataFrame | None = None, save_path: str | Path | None = None) -> pd.DataFrame:
    if df is None:
        df = load_raw_data()

    rng = np.random.default_rng(CONFIG.seed)
    all_steps = np.sort(df["step"].unique())
    if len(all_steps) < 3:
        raise ValueError("At least three time steps are required for injection and evaluation.")

    train_cutoff = int(len(all_steps) * 0.8)
    train_steps = all_steps[:train_cutoff]
    test_steps = all_steps[train_cutoff:]

    fraud_train_n = max(1, CONFIG.num_fraud_steps // 2)
    fraud_test_n = max(1, CONFIG.num_fraud_steps - fraud_train_n)
    fraud_steps_train = set(rng.choice(train_steps, size=min(fraud_train_n, len(train_steps)), replace=False).tolist())
    fraud_steps_test = set(rng.choice(test_steps, size=min(fraud_test_n, len(test_steps)), replace=False).tolist())
    fraud_steps = fraud_steps_train | fraud_steps_test

    remaining = [s for s in all_steps if s not in fraud_steps]
    demand_train_n = max(1, CONFIG.num_demand_steps // 2)
    demand_test_n = max(1, CONFIG.num_demand_steps - demand_train_n)
    demand_steps_train = set(rng.choice([s for s in train_steps if s not in fraud_steps], size=min(demand_train_n, len(train_steps) - len(fraud_steps_train)), replace=False).tolist())
    demand_steps_test = set(rng.choice([s for s in test_steps if s not in fraud_steps], size=min(demand_test_n, len(test_steps) - len(fraud_steps_test)), replace=False).tolist())
    demand_steps = demand_steps_train | demand_steps_test

    df_augmented = apply_injections(
        df,
        fraud_steps=list(fraud_steps),
        demand_steps=list(demand_steps),
        n_extra_fraud=CONFIG.n_extra_fraud,
        n_extra_demand=CONFIG.n_extra_demand,
        seed=CONFIG.seed,
    )

    bucket = create_bucket_features(df_augmented)
    bucket["label"] = bucket["step"].apply(lambda s: label_bucket(int(s), fraud_steps, demand_steps))

    # NEW: also save the row-level data (with ip_address/bin_number)
    # so the cluster report can look up WHO was involved in each cluster
    raw_save_path = Path(save_path).parent / "raw_augmented.csv" if save_path else OUTPUT_PATH.parent / "raw_augmented.csv"
    df_augmented.to_csv(raw_save_path, index=False)
    
    bucket["split"] = bucket["step"].apply(
    lambda s: "train" if s in train_steps else "test")

    if save_path is None:
        save_path = OUTPUT_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bucket.to_csv(save_path, index=False)
    return bucket


if __name__ == "__main__":
    bucket = build_labeled_buckets()

    print("\n" + "=" * 60)
    print("           FRAUD-SPIKE DETECTOR")
    print("=" * 60)

    print("\n[1] DATASET / SIMULATION")
    print("-" * 60)

    print(f"Total buckets : {len(bucket)}")

    counts = bucket["label"].value_counts()

    print("\nBucket distribution:")
    print(f"Normal          : {counts.get('normal', 0)}")
    print(f"Genuine demand  : {counts.get('genuine_demand', 0)}")
    print(f"Fraud attack    : {counts.get('fraud_attack', 0)}")

    summary = bucket.groupby("label")[
        ["txn_count", "unique_ratio", "avg_amount"]
    ].mean()

    print("\nAverage bucket statistics:")
    print(summary.round(2).to_string())