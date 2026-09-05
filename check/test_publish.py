import pandas as pd
from events.kafka_client import publish_fraud_event

fake_row = pd.Series({
    "step": 999,
    "txn_count": 4200,
    "unique_ratio": 0.12,
    "avg_amount": 320,
    "fraud_type_pred": "A",   # added to match the new signature
})
publish_fraud_event(fake_row)