from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd


TX_TYPES = ["transfer", "cash_out", "payment", "deposit", "withdrawal"]


@dataclass
class GeneratorConfig:
    num_accounts: int = 500
    num_transactions: int = 6000
    laundering_ratio: float = 0.12
    seed: int = 42


def _sample_normal_amount(rng: np.random.Generator, scale: float = 1.0) -> float:
    amount = rng.lognormal(mean=4.5, sigma=0.8) * scale
    return round(float(max(amount, 5.0)), 2)


def _generate_normal_transactions(accounts: list[str], total: int, rng: np.random.Generator) -> list[dict]:
    records: list[dict] = []
    start = pd.Timestamp("2025-01-01")
    for idx in range(total):
        sender, receiver = rng.choice(accounts, size=2, replace=False)
        timestamp = start + pd.Timedelta(minutes=int(idx * rng.integers(4, 30)))
        tx_type = rng.choice(TX_TYPES, p=[0.42, 0.12, 0.26, 0.1, 0.1])
        records.append(
            {
                "transaction_id": f"T{idx:07d}",
                "sender_id": sender,
                "receiver_id": receiver,
                "transaction_amount": _sample_normal_amount(rng),
                "timestamp": timestamp,
                "transaction_type": tx_type,
                "label": 0,
            }
        )
    return records


def _generate_laundering_transactions(
    accounts: list[str], start_idx: int, total: int, rng: np.random.Generator
) -> list[dict]:
    records: list[dict] = []
    start = pd.Timestamp("2025-02-15")
    suspicious_groups = []
    for _ in range(max(5, total // 20)):
        group = rng.choice(accounts, size=4, replace=False).tolist()
        suspicious_groups.append(group)

    current_idx = start_idx
    while len(records) < total:
        a, b, c, d = random.choice(suspicious_groups)
        base_time = start + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 40)))
        base_amount = _sample_normal_amount(rng, scale=5.0)
        pattern = [
            (a, b, base_amount * 0.98, "transfer", 0),
            (b, c, base_amount * 0.96, "transfer", 8),
            (c, d, base_amount * 0.94, "cash_out", 15),
        ]
        if rng.random() > 0.4:
            pattern.append((d, a, base_amount * 0.91, "deposit", 28))
        if rng.random() > 0.35:
            for step in range(3):
                pattern.append(
                    (
                        a,
                        rng.choice([b, c, d]),
                        round(base_amount / 6 + rng.uniform(10, 90), 2),
                        "payment",
                        35 + step * 4,
                    )
                )
        for sender, receiver, amount, tx_type, offset in pattern:
            if len(records) >= total:
                break
            records.append(
                {
                    "transaction_id": f"T{current_idx:07d}",
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "transaction_amount": round(float(amount), 2),
                    "timestamp": base_time + pd.Timedelta(minutes=offset),
                    "transaction_type": tx_type,
                    "label": 1,
                }
            )
            current_idx += 1
    return records


def generate_dataset(config: GeneratorConfig, output_path: str | Path) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    accounts = [f"ACC_{i:04d}" for i in range(config.num_accounts)]
    suspicious_total = int(config.num_transactions * config.laundering_ratio)
    normal_total = config.num_transactions - suspicious_total

    records = _generate_normal_transactions(accounts, normal_total, rng)
    records.extend(_generate_laundering_transactions(accounts, normal_total, suspicious_total, rng))

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df["transaction_amount"] = df["transaction_amount"].clip(lower=1.0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


if __name__ == "__main__":
    config = GeneratorConfig()
    path = Path("data") / "synthetic_aml_transactions.csv"
    df = generate_dataset(config, path)
    print(f"Saved {len(df)} transactions to {path}")
    print(df.head())
