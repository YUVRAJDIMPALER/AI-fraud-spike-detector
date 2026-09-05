from simulate.config import CONFIG
from simulate.build_buckets import build_labeled_buckets
from detect.stage2_classifier import train_fraud_type_classifier

# Save your normal settings so we can restore them after
original_fraud_steps = CONFIG.num_fraud_steps
original_demand_steps = CONFIG.num_demand_steps

try:
    # Temporarily boost fraud steps ONLY for this run, so each of the
    # 5 subtypes gets enough examples to train on reliably.
    CONFIG.num_fraud_steps = 1000
    CONFIG.num_demand_steps = 0  # don't need demand data for this model at all

    print(f"Building a larger fraud-only dataset ({CONFIG.num_fraud_steps} fraud steps)...")
    bucket = build_labeled_buckets()

    print("Training fraud-type classifier on this larger dataset...")
    train_fraud_type_classifier(bucket)

finally:
    # ALWAYS restore original settings, even if something above fails
    CONFIG.num_fraud_steps = original_fraud_steps
    CONFIG.num_demand_steps = original_demand_steps
    print(f"Restored config: fraud_steps={CONFIG.num_fraud_steps}, demand_steps={CONFIG.num_demand_steps}")