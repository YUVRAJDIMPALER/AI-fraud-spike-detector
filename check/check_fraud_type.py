from simulate.build_buckets import build_labeled_buckets

bucket = build_labeled_buckets()
print("Columns:", bucket.columns.tolist())
print()
print("fraud_type value counts among fraud_attack buckets:")
print(bucket[bucket["label"] == "fraud_attack"]["fraud_type"].value_counts(dropna=False))