from __future__ import annotations

import os
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stage2_model.pkl"


def prepare_features(bucket: pd.DataFrame) -> pd.DataFrame:
    required = {"txn_count", "unique_senders", "avg_amount", "total_amount", "unique_ratio", "amount_ratio", "fraud_rate", "label"}
    missing = sorted(required - set(bucket.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    label_map = {"normal": 0, "genuine_demand": 1, "fraud_attack": 2}
    features = bucket[[
        "txn_count", "unique_senders", "avg_amount", "total_amount",
        "unique_ratio", "amount_ratio"  # NEW ,"top_ip_share", "top_bin_share",
    ]].copy()    # //, "fraud_rate"
    labels = bucket["label"].map(label_map)
    return features, labels


def train_classifier(bucket: pd.DataFrame, model_type: str = "random_forest"):
    X, y = prepare_features(bucket)
    #stratify = y if y.value_counts().min() >= 2 else None
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    X_train = X
    y_train = y

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, multi_class="auto")
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",  # needed for multi-class to suppress a warning
        )
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    # Train model
    model.fit(X_train, y_train)


    

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\n[3] STAGE 2 - MODEL TRAINING")
    print("-" * 60)
    print(f"Training samples : {len(X_train)}")
    print("Model            : XG boost Classifier" if model_type == "xgboost" else model.__class__.__name__)
    print(f"Features         : {list(X_train.columns)}")
    print(f"Model saved to   : {MODEL_PATH}")

    return model


def predict_stage2(bucket: pd.DataFrame, model=None) -> pd.DataFrame:
    if model is None:
        model = joblib.load(MODEL_PATH)

    X, _ = prepare_features(bucket)
    pred = model.predict(X)
    classes = {0: "normal", 1: "genuine_demand", 2: "fraud_attack"}
    bucket = bucket.copy()
    bucket["stage2_pred"] = [classes.get(int(p), "normal") for p in pred]
    return bucket


FRAUD_TYPE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_type_model.pkl"

def train_fraud_type_classifier(bucket: pd.DataFrame):
    fraud_only = bucket[bucket["label"] == "fraud_attack"].dropna(subset=["fraud_type"])

    features = fraud_only[[
        "txn_count", "unique_senders", "avg_amount", "total_amount",
        "unique_ratio", "amount_ratio", "top_ip_share", "top_bin_share",
    ]].copy()

    type_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    labels = fraud_only["fraud_type"].map(type_map)
    """
    Train a classifier to predict the type of fraud based on the features.
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(features, labels)
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(features, labels)


    FRAUD_TYPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FRAUD_TYPE_MODEL_PATH)
    print(f"Fraud-type classifier saved to {FRAUD_TYPE_MODEL_PATH}")
    return model


def predict_fraud_type(bucket: pd.DataFrame, model=None) -> pd.Series:
    if model is None:
        model = joblib.load(FRAUD_TYPE_MODEL_PATH)

    features = bucket[[
        "txn_count", "unique_senders", "avg_amount", "total_amount",
        "unique_ratio", "amount_ratio", "top_ip_share", "top_bin_share",
    ]].copy()

    type_map_inv = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    preds = model.predict(features)
    return pd.Series([type_map_inv[p] for p in preds], index=bucket.index)

