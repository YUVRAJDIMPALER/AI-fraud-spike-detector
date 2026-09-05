import json
from pathlib import Path

from simulate.build_buckets import build_labeled_buckets
from evaluate.split import time_split
from detect.stage2_classifier import predict_fraud_type
from sklearn.metrics import classification_report, confusion_matrix


METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_type_metrics.json"


def evaluate_fraud_type() -> dict:
	bucket = build_labeled_buckets()
	_, test = time_split(bucket, test_fraction=0.2)

	fraud_test = test[test["label"] == "fraud_attack"].dropna(subset=["fraud_type"])
	labels = ["A", "B", "C", "D", "E"]
	preds = predict_fraud_type(fraud_test)
	report = classification_report(
		fraud_test["fraud_type"],
		preds,
		labels=labels,
		output_dict=True,
		zero_division=0,
	)
	matrix = confusion_matrix(
		fraud_test["fraud_type"],
		preds,
		labels=labels,
	)
	result = {
		"label": "Fraud Type Classifier",
		"labels": labels,
		"accuracy": round(report["accuracy"], 4),
		"confusion_matrix": matrix.tolist(),
		"classification_report": report,
	}
	METRICS_PATH.write_text(json.dumps(result, indent=2))
	return result


if __name__ == "__main__":
	result = evaluate_fraud_type()
	print(json.dumps(result["classification_report"], indent=2))
	print(f"Saved fraud-type evaluation to {METRICS_PATH}")