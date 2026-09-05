import pandas as pd

# load your row-level data with fraud types (if you saved fraud_type as a column)
# and your bucket-level predictions
raw_df = pd.read_csv("data/processed/raw_augmented.csv", low_memory=False)
bucket_df = pd.read_csv("data/processed/buckets_labeled.csv")  # or wherever your labeled buckets are saved

# find genuine_demand buckets that got predicted as fraud_attack
# (you'll need to merge in stage2_pred from your evaluation run - or just
# manually list the step numbers from your confusion matrix / detection output)

# for now, let's just check: for genuine_demand buckets, what does their
# unique_ratio / avg_amount / top_ip_share look like vs fraud_attack buckets?
demand_buckets = bucket_df[bucket_df["label"] == "genuine_demand"]
fraud_buckets = bucket_df[bucket_df["label"] == "fraud_attack"]

print("Genuine demand stats:")
print(demand_buckets[["unique_ratio", "avg_amount", "top_ip_share", "top_bin_share"]].describe())

print("\nFraud attack stats:")
print(fraud_buckets[["unique_ratio", "avg_amount", "top_ip_share", "top_bin_share"]].describe())