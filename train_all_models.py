"""
Trains all three Stage 2 model candidates (Random Forest, Gradient Boosting,
XGBoost) on the SAME train/test split, evaluates each on the held-out test
set, and saves:

  models/stage2_random_forest.pkl
  models/stage2_gradient_boosting.pkl
  models/stage2_xgboost.pkl
  models/model_metrics.json   <- real, freshly-measured comparison numbers

Run this once (or whenever you regenerate your simulated data) so the
dashboard's model picker always reflects your actual current data, not
stale numbers from an earlier session.

Usage:
    python train_all_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from detect.pipeline import run_detection_pipeline
from detect.stage2_classifier import prepare_features
from evaluate.metrics import fraud_catch_metrics
from evaluate.split import time_split
from simulate.build_buckets import build_labeled_buckets

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_BUILDERS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        eval_metric="mlogloss",
    ),
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def main() -> None:
    bucket = build_labeled_buckets()
    train, test = time_split(bucket, test_fraction=0.2)
    X_train, y_train = prepare_features(train)

    results: dict = {}

    for model_id, builder in MODEL_BUILDERS.items():
        print(f"\nTraining {MODEL_LABELS[model_id]} ...")
        model = builder()
        model.fit(X_train, y_train)

        model_path = MODELS_DIR / f"stage2_{model_id}.pkl"
        joblib.dump(model, model_path)

        # Evaluate on the held-out test split, using the same pipeline
        # logic your live system uses (Stage 1 + Stage 2 + final_flag).
        test_pred = run_detection_pipeline(test.copy(), model=model)
        fraud_metrics = fraud_catch_metrics(test_pred, true_col="label", flag_col="final_flag")
        report_labels = ["normal", "genuine_demand", "fraud_attack"]
        three_way = classification_report(
            test_pred["label"],
            test_pred["stage2_pred"],
            labels=report_labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(
            test_pred["label"],
            test_pred["stage2_pred"],
            labels=report_labels,
        )

        results[model_id] = {
            "label": MODEL_LABELS[model_id],
            "accuracy": round(three_way["accuracy"], 4),
            "fraud_precision": round(fraud_metrics["precision"], 4),
            "fraud_recall": round(fraud_metrics["recall"], 4),
            "fraud_f1": round(fraud_metrics["f1"], 4),
            "genuine_demand_recall": round(three_way.get("genuine_demand", {}).get("recall", 0.0), 4),
            "fraud_catch": {
                "precision": round(fraud_metrics["precision"], 4),
                "recall": round(fraud_metrics["recall"], 4),
                "f1": round(fraud_metrics["f1"], 4),
                "counts": fraud_metrics["counts"],
                "cost": round(fraud_metrics["cost"], 4),
            },
            "evaluation": {
                "labels": report_labels,
                "confusion_matrix": matrix.tolist(),
                "classification_report": three_way,
            },
        }

        print(
            f"  accuracy={results[model_id]['accuracy']:.3f}  "
            f"fraud_recall={results[model_id]['fraud_recall']:.3f}  "
            f"fraud_precision={results[model_id]['fraud_precision']:.3f}"
        )

    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"default": "xgboost", "models": results}, f, indent=2)

    print(f"\nSaved model comparison to {metrics_path}")
    print("Restart your API server so it picks up the new metrics and models.")


if __name__ == "__main__":
    main()
