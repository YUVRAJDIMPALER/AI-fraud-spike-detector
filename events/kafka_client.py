from kafka import KafkaProducer
import json
import uuid
from datetime import datetime, timezone

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def publish_fraud_event(row):
    fraud_type = row["fraud_type_pred"]
    event = {
        "event_id": str(uuid.uuid4()),
        "step": int(row["step"]),
        "fraud_type": fraud_type,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "features": {
            "txn_count": int(row["txn_count"]),
            "unique_ratio": float(row["unique_ratio"]),
            "avg_amount": float(row["avg_amount"]),
        },
    }
    topic = f"fraud.type{fraud_type}.problem"
    producer.send(topic, value=event)
    producer.flush()
    print(f"Published event {event['event_id']} to {topic}")