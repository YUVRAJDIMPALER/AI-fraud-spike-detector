CREATE TABLE IF NOT EXISTS transaction_bucket (
    step INTEGER PRIMARY KEY,
    txn_count INTEGER,
    unique_senders INTEGER,
    avg_amount DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    unique_ratio DOUBLE PRECISION,
    fraud_rate DOUBLE PRECISION,
    label TEXT
);

CREATE TABLE IF NOT EXISTS risk_event (
    id SERIAL PRIMARY KEY,
    step INTEGER,
    signal_type TEXT,
    risk_score DOUBLE PRECISION,
    reason_codes TEXT[]
);

CREATE TABLE IF NOT EXISTS review_outcome (
    id SERIAL PRIMARY KEY,
    step INTEGER,
    reviewer_decision TEXT,
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
