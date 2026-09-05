from kafka import KafkaProducer, KafkaConsumer
import json, time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)
producer.send("test.topic", {"hello": "world"})
producer.flush()
print("Sent a test message.")