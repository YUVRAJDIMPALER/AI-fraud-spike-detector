from __future__ import annotations

import joblib
import pandas as pd

from detect.stage1_anomaly import stage1_flag
from detect.stage2_classifier import MODEL_PATH, predict_stage2, predict_fraud_type
from simulate.build_buckets import build_labeled_buckets


def run_detection_pipeline(
    bucket: pd.DataFrame,
    model=None
) -> pd.DataFrame:

    # Stage 1: detect unusual activity
    flagged = stage1_flag(bucket)

    # Load trained model if one wasn't provided
    if model is None:
        model = joblib.load(MODEL_PATH)

    # Stage 2: classify the event
    predicted = predict_stage2(
        flagged,
        model=model
    )

    # Final decision
    predicted["final_flag"] = (
        predicted["stage2_pred"].eq("fraud_attack").astype(int)
    )

    predicted["genuine_demand_flag"] = (
        predicted["stage2_pred"].eq("genuine_demand").astype(int)
    )

    # NEW: for every bucket flagged as fraud, predict WHICH fraud type it is
    predicted["fraud_type_pred"] = None
    fraud_rows = predicted[predicted["stage2_pred"] == "fraud_attack"]
    if not fraud_rows.empty:
        predicted.loc[fraud_rows.index, "fraud_type_pred"] = predict_fraud_type(fraud_rows)

    # Explain why the bucket was flagged
    predicted["reason_codes"] = predicted.apply(
        lambda row: {
            "anomaly": int(row["anomaly_flag"]),
            "type": row["stage2_pred"],
            "fraud_type": row["fraud_type_pred"],
            "label": row["label"]
        },
        axis=1
    )

    return predicted


if __name__ == "__main__":

    bucket = build_labeled_buckets()

    result = run_detection_pipeline(bucket)

    print("\n[5] DETECTION RESULTS")
    print("-" * 60)

    print(result[
        [
            "step",
            "anomaly_flag",
            "stage2_pred",
            "fraud_type_pred",
            "final_flag"
        ]
    ].tail(20).to_string(index=False))