import sys
from kafka import KafkaConsumer, KafkaProducer
import json

DEPARTMENT_NAMES = {
    "A": "Cyber-Crime Unit",
    "B": "AML / Mule Network Team",
    "C": "Card Issuer Relations",
    "D": "Senior Fraud Ops (Escalation)",
    "E": "Account Security Team",
}

def run_department_consumer(fraud_type):
    dept_name = DEPARTMENT_NAMES.get(fraud_type, "Unknown Department")

    consumer = KafkaConsumer(
        f"fraud.type{fraud_type}.problem",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"[{dept_name}] Listening on fraud.type{fraud_type}.problem ...")
    for message in consumer:
        event = message.value
        print(f"[{dept_name}] Received event: {event['event_id']} (step {event['step']})")

        producer.send(f"fraud.type{fraud_type}.acknowledgement", value={
            "event_id": event["event_id"], "status": "under_review", "department": dept_name,
        })
        print(f"[{dept_name}]   -> Acknowledged")

        # Placeholder decision logic - in a real system, a human reviews here
        decision = "flagged"
        producer.send(f"fraud.type{fraud_type}.action", value={
            "event_id": event["event_id"], "decision": decision,
        })
        print(f"[{dept_name}]   -> Action: {decision}")
        producer.flush()

if __name__ == "__main__":
    fraud_type = sys.argv[1] if len(sys.argv) > 1 else "A"
    run_department_consumer(fraud_type)