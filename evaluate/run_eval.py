from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

from detect.pipeline import run_detection_pipeline
from evaluate.metrics import fraud_catch_metrics
from evaluate.split import time_split
from evaluate.plots import generate_cluster_action_report
from evaluate.plots import plot_demand_vs_fraud, plot_fraud_clusters, plot_class_metrics, plot_confusion
from simulate.build_buckets import build_labeled_buckets
from evaluate.plots import plot_stage1_anomalies, plot_stage2_classification

def run_evaluation() -> dict:
    bucket = build_labeled_buckets()
    train, test = time_split(bucket, test_fraction=0.2)

    # NEW: load the row-level data (has ip_address, bin_number columns)
    # this is separate from `bucket`, which is the aggregated-per-step data
    raw_df = pd.read_csv("data/processed/raw_augmented.csv")

    model = joblib.load("models/stage2_model.pkl")
    test_pred = run_detection_pipeline(test, model=model)

    # pass the FULL bucket (to find clusters across all steps) and the
    # row-level raw_df (to look up which IP/BIN was behind each cluster)
    generate_cluster_action_report(bucket, raw_df, max_gap=10)

    # ---------- REPORT 1: Did we catch the fraud? ----------
    fraud_metrics = fraud_catch_metrics(test_pred, true_col="label", flag_col="final_flag")

    print("\n[4] FRAUD CATCH PERFORMANCE (fraud_attack only)")
    print("-" * 60)
    print(f"Precision          : {fraud_metrics['precision']:.4f}")
    print(f"Recall             : {fraud_metrics['recall']:.4f}")
    print(f"F1 Score           : {fraud_metrics['f1']:.4f}")
    print("\nConfusion counts:")
    print(f"True Positives     : {fraud_metrics['counts']['TP']}")
    print(f"False Positives    : {fraud_metrics['counts']['FP']}")
    print(f"True Negatives     : {fraud_metrics['counts']['TN']}")
    print(f"False Negatives    : {fraud_metrics['counts']['FN']}")
    print(f"\nTotal Cost         : {fraud_metrics['cost']:.2f}")

    # ---------- REPORT 2: Can we tell all 3 types apart? ----------
    print("\n[5] THREE-WAY CLASSIFICATION (normal vs genuine_demand vs fraud_attack)")
    print("-" * 60)
    report = classification_report(
        test_pred["label"], test_pred["stage2_pred"], zero_division=0
    )
    print(report)

    #plot_stage1_anomalies(test_pred)
    #plot_stage2_classification(test_pred)
    #plot_fraud_clusters(bucket, max_gap=10)
    plot_confusion(test_pred["label"], test_pred["stage2_pred"])

    return {"fraud_metrics": fraud_metrics}


if __name__ == "__main__":
    run_evaluation()