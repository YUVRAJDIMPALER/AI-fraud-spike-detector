def explain_fallback(step: int, txn_count: float, unique_ratio: float, avg_amount: float, label: str, stage2_pred: str) -> str:
    if stage2_pred == "fraud_attack":
        return (
            f"At step {step}, the system observed a concentrated burst of activity with {txn_count} transactions, "
            f"low sender diversity (unique_ratio={unique_ratio:.3f}), and unusually small amounts ({avg_amount:.2f}). "
            "This pattern is consistent with a fraud attack or card-testing campaign."
        )
    if stage2_pred == "genuine_demand":
        return (
            f"At step {step}, the system saw a volume increase with {txn_count} transactions and a broad customer base "
            f"(unique_ratio={unique_ratio:.3f}). The transaction size is normal for demand surges, suggesting genuine customer activity."
        )
    return (
        f"At step {step}, behavior stayed near baseline with {txn_count} transactions and moderate sender diversity "
        f"(unique_ratio={unique_ratio:.3f}). This looks normal and does not require urgent intervention."
    )
