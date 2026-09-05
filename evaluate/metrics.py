from __future__ import annotations

import pandas as pd


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def precision_recall_f1(counts: dict) -> dict:
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_cost(counts: dict, fp_cost: float = 1.0, fn_cost: float = 5.0) -> float:
    return counts["FP"] * fp_cost + counts["FN"] * fn_cost


def evaluate_predictions(df: pd.DataFrame, true_col: str = "label", pred_col: str = "pred") -> dict:
    positives = df[true_col].map({"fraud_attack": 1, "genuine_demand": 1, "normal": 0})
    predictions = df[pred_col].map({"fraud_attack": 1, "genuine_demand": 1, "normal": 0})

    counts = confusion_counts(positives, predictions)
    metrics = precision_recall_f1(counts)
    metrics.update({"counts": counts, "cost": compute_cost(counts)})
    return metrics


def fraud_catch_metrics(df: pd.DataFrame, true_col: str = "label", flag_col: str = "final_flag") -> dict:
    """
    Report card #1: 'Did we catch the fraud?'

    This ONLY cares about fraud_attack. Genuine_demand is treated as
    'nothing to catch here' — same as normal. This gives us a clean,
    honest number for fraud-catching, not mixed up with demand spikes.
    """
    # true answer: 1 if this bucket is REALLY a fraud attack, else 0
    y_true = (df[true_col] == "fraud_attack").astype(int)

    # our guess: 1 if we flagged it, else 0
    y_pred = df[flag_col].astype(int)

    counts = confusion_counts(y_true, y_pred)
    metrics = precision_recall_f1(counts)
    metrics.update({"counts": counts, "cost": compute_cost(counts)})
    return metrics