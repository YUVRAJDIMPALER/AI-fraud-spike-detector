from __future__ import annotations

import numpy as np
import pandas as pd

from simulate.config import CONFIG


def _random_ip(rng) -> str:
    return f"{rng.integers(1,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}"


def _random_bin(rng) -> str:
    return str(rng.integers(100000, 999999))


def _base_row(step, txn_type, amount, sender, dest, ip, bin_no, is_fraud, fraud_type=None):
    return {
        "step": int(step),
        "type": txn_type,
        "amount": float(amount),
        "nameOrig": sender,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "nameDest": dest,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": is_fraud,
        "ip_address": ip,
        "bin_number": bin_no,
        "fraud_type": fraud_type   # NEW
    }


def _pick_fraud_ip(rng, core_ips, purity=0.75):
    """
    Most fraud transactions reuse the small 'core' set of IPs (that's the
    real signal). But some fraction come from a totally random IP instead -
    real attackers rotate through proxies/VPNs sometimes, so it's not a
    perfectly clean giveaway.
    """
    if rng.random() < purity:
        return rng.choice(core_ips)
    return _random_ip(rng)


def _pick_fraud_bin(rng, core_bins, purity=0.75):
    if rng.random() < purity:
        return rng.choice(core_bins)
    return _random_bin(rng)


def inject_fraud_type_a(step, n_extra, rng):
    """Type A: many senders -> one destination (classic card-testing bot farm)."""
    core_ips = [_random_ip(rng) for _ in range(CONFIG.n_fraud_ips)]
    core_bins = [_random_bin(rng) for _ in range(CONFIG.n_fraud_bins)]
    dest = "FRAUD_DEST_1"
    rows = []
    for _ in range(n_extra):
        rows.append(_base_row(
            step, "TRANSFER", rng.uniform(10, 1500),
            f"FRAUD_SRC_{rng.integers(0,5)}", dest,
            _pick_fraud_ip(rng, core_ips), _pick_fraud_bin(rng, core_bins), 1,
             fraud_type="A",
        ))
    return pd.DataFrame(rows)


def inject_fraud_type_b(step, n_extra, rng):
    """Type B: few senders -> many destinations (mule / takeover network)."""
    core_ips = [_random_ip(rng) for _ in range(CONFIG.n_fraud_ips)]
    core_bins = [_random_bin(rng) for _ in range(CONFIG.n_fraud_bins)]
    senders = [f"FRAUD_SRC_{i}" for i in range(3)]
    rows = []
    for _ in range(n_extra):
        rows.append(_base_row(
            step, "TRANSFER", rng.uniform(50, 800),
            rng.choice(senders), f"MULE_DEST_{rng.integers(0,200)}",
            _pick_fraud_ip(rng, core_ips), _pick_fraud_bin(rng, core_bins), 1,
            fraud_type="B",
        ))
    return pd.DataFrame(rows)


def inject_fraud_type_c(step, n_extra, rng):
    """Type C: many small transactions (card-testing probe amounts)."""
    core_ips = [_random_ip(rng) for _ in range(CONFIG.n_fraud_ips)]
    core_bins = [_random_bin(rng) for _ in range(CONFIG.n_fraud_bins)]
    rows = []
    for _ in range(n_extra):
        rows.append(_base_row(
            step, "PAYMENT", rng.uniform(1, 20),
            f"FRAUD_SRC_{rng.integers(0,10)}", "FRAUD_DEST_2",
            _pick_fraud_ip(rng, core_ips), _pick_fraud_bin(rng, core_bins), 1,
            fraud_type="C",
        ))
    return pd.DataFrame(rows)


def inject_fraud_type_d(step, n_extra, rng):
    """Type D: sudden high-value transfers (account drain). Fewer rows, big amounts."""
    core_ips = [_random_ip(rng) for _ in range(CONFIG.n_fraud_ips)]
    core_bins = [_random_bin(rng) for _ in range(CONFIG.n_fraud_bins)]
    small_n = max(3, n_extra // 10)
    rows = []
    for _ in range(small_n):
        rows.append(_base_row(
            step, "TRANSFER", rng.uniform(50000, 200000),
            f"FRAUD_SRC_{rng.integers(0,3)}", "FRAUD_DEST_3",
            _pick_fraud_ip(rng, core_ips), _pick_fraud_bin(rng, core_bins), 1,
            fraud_type="D",
        ))
    return pd.DataFrame(rows)


def inject_fraud_type_e(step, n_extra, rng):
    """Type E: rapid repeated transactions (same sender/dest, compromised session)."""
    core_ips = [_random_ip(rng) for _ in range(max(1, CONFIG.n_fraud_ips - 1))]
    core_bins = [_random_bin(rng) for _ in range(1)]
    sender = "FRAUD_SRC_SESSION"
    dest = "FRAUD_DEST_4"
    rows = []
    for _ in range(n_extra):
        rows.append(_base_row(
            step, "TRANSFER", rng.uniform(100, 300),
            sender, dest, _pick_fraud_ip(rng, core_ips), _pick_fraud_bin(rng, core_bins), 1,
            fraud_type="E",
        ))
    return pd.DataFrame(rows)


FRAUD_TYPES = {
    "A": inject_fraud_type_a,
    "B": inject_fraud_type_b,
    "C": inject_fraud_type_c,
    "D": inject_fraud_type_d,
    "E": inject_fraud_type_e,
}


# A shared pool of "neighborhood" IPs and BINs that genuine customers
# realistically reuse (same city ISP, same popular banks) - NOT unique
# to any one attack, so it looks similar to background noise, not a signal.
_DEMAND_IP_POOL = None
_DEMAND_BIN_POOL = None


def _get_demand_pools(rng):
    global _DEMAND_IP_POOL, _DEMAND_BIN_POOL
    if _DEMAND_IP_POOL is None:
        _DEMAND_IP_POOL = [_random_ip(rng) for _ in range(40)]
        _DEMAND_BIN_POOL = [_random_bin(rng) for _ in range(15)]
    return _DEMAND_IP_POOL, _DEMAND_BIN_POOL


def inject_demand_spike(df, step, n_extra=150, seed=None):
    rng = np.random.default_rng(seed)
    ip_pool, bin_pool = _get_demand_pools(rng)

    # a pool of "repeat customers" - about 30% of transactions will
    # reuse one of these, instead of always being a brand new customer
    repeat_customers = [f"CUST_{rng.integers(0, 100000)}" for _ in range(20)]

    rows = []
    for _ in range(n_extra):
        if rng.random() < 0.3:
            sender = rng.choice(repeat_customers)
        else:
            sender = f"CUST_{rng.integers(0, 100000)}"

        rows.append(_base_row(
            step, "PAYMENT", rng.uniform(200, 3000),
            sender, "MERCHANT_SALE_1",
            rng.choice(ip_pool), rng.choice(bin_pool), 0,
        ))
    return pd.DataFrame(rows)


def apply_injections(df, fraud_steps, demand_steps, n_extra_fraud=150, n_extra_demand=150, seed=42):
    rng = np.random.default_rng(seed)
    injected_frames = [df]

    type_names = list(FRAUD_TYPES.keys())
    for s in fraud_steps:
        chosen_type = rng.choice(type_names)
        fn = FRAUD_TYPES[chosen_type]
        injected_frames.append(fn(s, n_extra_fraud, rng))

    for s in demand_steps:
        injected_frames.append(inject_demand_spike(df, s, n_extra=n_extra_demand, seed=int(rng.integers(0, 1_000_000))))

    return pd.concat(injected_frames, ignore_index=True)