PROMPT_TEMPLATE = """
You are a fraud and demand analyst.
Explain the observed bucket-level anomaly in plain English.

Context:
- step: {step}
- txn_count: {txn_count}
- unique_ratio: {unique_ratio}
- avg_amount: {avg_amount}
- label: {label}
- stage2_pred: {stage2_pred}

Write a concise explanation with:
1. likely cause
2. why it looks suspicious or normal
3. recommended action
"""
