from simulate.build_buckets import build_labeled_buckets
from detect.pipeline import run_detection_pipeline
from events.kafka_client import publish_fraud_event
import time

bucket = build_labeled_buckets()
result = run_detection_pipeline(bucket)

fraud_events = result[result["final_flag"] == 1]
print(f"Found {len(fraud_events)} fraud events to publish.\n")

for _, row in fraud_events.iterrows():
    publish_fraud_event(row)
    time.sleep(0.5)