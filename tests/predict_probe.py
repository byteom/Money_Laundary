from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
import sys
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment.app import app


@dataclass
class ProbeCase:
    case_id: str
    category: str
    payload: dict


def _base_payload() -> dict:
    return {
        "sender_id": "ACC_0001",
        "receiver_id": "ACC_0002",
        "transaction_amount": 100.0,
        "timestamp": "2025-01-15T10:30:00Z",
        "transaction_type": "transfer",
    }


def build_cases(target_count: int = 120) -> list[ProbeCase]:
    cases: list[ProbeCase] = []

    # 1) Amount sweep (12)
    for i, amount in enumerate([5, 20, 50, 100, 200, 500, 1_000, 2_500, 5_000, 9_900, 25_000, 60_000], start=1):
        payload = _base_payload()
        payload["transaction_amount"] = float(amount)
        cases.append(ProbeCase(f"amount_{i:02d}", "amount", payload))

    # 2) Type sweep (5)
    for tx_type in ["transfer", "payment", "cash_out", "deposit", "withdrawal"]:
        payload = _base_payload()
        payload["transaction_type"] = tx_type
        cases.append(ProbeCase(f"type_{tx_type}", "type", payload))

    # 3) Hour sweep (8)
    for hour in [0, 3, 6, 9, 12, 15, 18, 23]:
        payload = _base_payload()
        payload["timestamp"] = f"2025-01-15T{hour:02d}:10:00Z"
        cases.append(ProbeCase(f"hour_{hour:02d}", "hour", payload))

    # 4) Known account combinations and direction swap (8)
    known_pairs = [
        ("ACC_0001", "ACC_0002"),
        ("ACC_0002", "ACC_0001"),
        ("ACC_0010", "ACC_0011"),
        ("ACC_0011", "ACC_0010"),
        ("ACC_0100", "ACC_0101"),
        ("ACC_0101", "ACC_0100"),
        ("ACC_0200", "ACC_0300"),
        ("ACC_0300", "ACC_0200"),
    ]
    for idx, (sender, receiver) in enumerate(known_pairs, start=1):
        payload = _base_payload()
        payload["sender_id"] = sender
        payload["receiver_id"] = receiver
        payload["transaction_amount"] = float(500 + idx * 120)
        cases.append(ProbeCase(f"known_pair_{idx:02d}", "known_pair", payload))

    # 5) Unknown account scenarios (7)
    unknown_pairs = [
        ("UNK_A_1", "UNK_B_1", 75),
        ("UNK_A_2", "UNK_B_2", 500),
        ("UNK_A_3", "UNK_B_3", 9000),
        ("UNK_A_4", "ACC_0001", 1500),
        ("ACC_0001", "UNK_B_4", 1800),
        ("UNK_A_5", "UNK_B_5", 25000),
        ("UNK_A_6", "UNK_A_6", 3000),
    ]
    for idx, (sender, receiver, amount) in enumerate(unknown_pairs, start=1):
        payload = _base_payload()
        payload["sender_id"] = sender
        payload["receiver_id"] = receiver
        payload["transaction_amount"] = float(amount)
        payload["timestamp"] = f"2025-01-{15 + idx:02d}T0{idx % 10}:22:00Z"
        cases.append(ProbeCase(f"unknown_{idx:02d}", "unknown", payload))

    # 6) Risk boundary nudges (10)
    boundary_amounts = [90, 95, 99, 100, 101, 4900, 5000, 5100, 9900, 10000]
    for idx, amount in enumerate(boundary_amounts, start=1):
        payload = _base_payload()
        payload["transaction_amount"] = float(amount)
        payload["timestamp"] = f"2025-01-2{idx % 10}T14:{idx:02d}:00Z"
        cases.append(ProbeCase(f"boundary_{idx:02d}", "boundary", payload))

    # 7) Day-of-week sweep (7)
    day_samples = [
        "2025-01-13T08:20:00Z",  # Mon
        "2025-01-14T08:20:00Z",  # Tue
        "2025-01-15T08:20:00Z",  # Wed
        "2025-01-16T08:20:00Z",  # Thu
        "2025-01-17T08:20:00Z",  # Fri
        "2025-01-18T08:20:00Z",  # Sat
        "2025-01-19T08:20:00Z",  # Sun
    ]
    for idx, timestamp in enumerate(day_samples, start=1):
        payload = _base_payload()
        payload["timestamp"] = timestamp
        payload["transaction_amount"] = float(250 + idx * 90)
        payload["transaction_type"] = ["transfer", "payment", "cash_out", "deposit", "withdrawal", "transfer", "payment"][idx - 1]
        cases.append(ProbeCase(f"dow_{idx:02d}", "day_of_week", payload))

    # 8) Burst sequence simulation (24)
    burst_sender = "ACC_0333"
    burst_receiver = "ACC_0444"
    burst_start = pd.Timestamp("2025-02-05T10:00:00Z")
    for idx in range(24):
        payload = _base_payload()
        payload["sender_id"] = burst_sender
        payload["receiver_id"] = burst_receiver
        payload["transaction_amount"] = float(800 + (idx % 6) * 450)
        payload["timestamp"] = (burst_start + pd.Timedelta(minutes=idx * 10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["transaction_type"] = ["transfer", "cash_out", "payment", "transfer", "withdrawal", "deposit"][idx % 6]
        cases.append(ProbeCase(f"burst_{idx + 1:02d}", "burst", payload))

    # 9) Mixed risk matrix (36)
    matrix_amounts = [45.0, 320.0, 2100.0, 9700.0, 26000.0, 68000.0]
    matrix_hours = [1, 6, 11, 16, 21, 23]
    matrix_types = ["payment", "transfer", "cash_out", "deposit", "withdrawal", "transfer"]
    for idx in range(39):
        payload = _base_payload()
        amount = matrix_amounts[idx % len(matrix_amounts)]
        hour = matrix_hours[idx % len(matrix_hours)]
        tx_type = matrix_types[idx % len(matrix_types)]
        payload["sender_id"] = f"ACC_{(idx * 7 + 10) % 500:04d}"
        payload["receiver_id"] = f"ACC_{(idx * 13 + 20) % 500:04d}"
        payload["transaction_amount"] = float(amount)
        payload["timestamp"] = f"2025-03-{(idx % 28) + 1:02d}T{hour:02d}:{(idx * 3) % 60:02d}:00Z"
        payload["transaction_type"] = tx_type
        cases.append(ProbeCase(f"matrix_{idx + 1:02d}", "matrix", payload))

    if len(cases) < target_count:
        raise RuntimeError(f"Expected at least {target_count} cases, got {len(cases)}")
    return cases[:target_count]


def run_probe(output_prefix: str, case_count: int = 120) -> dict:
    out_dir = Path("artifacts") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{output_prefix}_probe_results.csv"
    summary_path = out_dir / f"{output_prefix}_probe_summary.json"

    rows = []
    errors = []

    with app.test_client() as client:
        for case in build_cases(case_count):
            resp = client.post("/predict", json=case.payload)
            if resp.status_code != 200:
                errors.append({
                    "case_id": case.case_id,
                    "status": resp.status_code,
                    "response": resp.get_data(as_text=True),
                })
                continue

            data = resp.get_json()
            explainability = data.get("explainability", {}) if isinstance(data, dict) else {}
            confidence = explainability.get("confidence", {}) if isinstance(explainability, dict) else {}
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "sender_id": case.payload["sender_id"],
                    "receiver_id": case.payload["receiver_id"],
                    "amount": case.payload["transaction_amount"],
                    "timestamp": case.payload["timestamp"],
                    "transaction_type": case.payload["transaction_type"],
                    "baseline_probability": float(data["baseline_probability"]),
                    "graphsage_probability": float(data["graphsage_probability"]),
                    "tgn_probability": float(data["tgn_probability"]),
                    "ensemble_probability": float(data["ensemble_probability"]),
                    "risk_classification": data["risk_classification"],
                    "explainability_present": int("explainability" in data),
                    "confidence_score": float(confidence.get("score", 0.0)),
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    ensemble_values = [row["ensemble_probability"] for row in rows]
    unique_ensemble = sorted(set(ensemble_values))

    per_category_variance = {}
    for category in sorted(set(row["category"] for row in rows)):
        vals = [row["ensemble_probability"] for row in rows if row["category"] == category]
        per_category_variance[category] = {
            "count": len(vals),
            "unique": len(set(vals)),
            "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
            "mean": round(mean(vals), 6) if vals else 0.0,
        }

    summary = {
        "cases_attempted": case_count,
        "cases_successful": len(rows),
        "errors": errors,
        "unique_ensemble_count": len(unique_ensemble),
        "unique_ensemble_values": unique_ensemble[:50],
        "ensemble_std": round(pstdev(ensemble_values), 6) if len(ensemble_values) > 1 else 0.0,
        "risk_class_counts": {
            cls: sum(1 for row in rows if row["risk_classification"] == cls)
            for cls in sorted(set(row["risk_classification"] for row in rows))
        },
        "explainability_coverage": round(
            sum(row.get("explainability_present", 0) for row in rows) / max(len(rows), 1),
            4,
        ),
        "per_category": per_category_variance,
        "csv_path": str(csv_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 100+ case /predict diversity probe.")
    parser.add_argument("--prefix", default="before_fix", help="Output prefix under artifacts/reports")
    parser.add_argument("--count", type=int, default=120, help="Number of deterministic scenarios to run")
    args = parser.parse_args()

    summary = run_probe(args.prefix, case_count=args.count)
    print(json.dumps(summary, indent=2))
