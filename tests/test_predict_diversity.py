from __future__ import annotations

from collections import Counter
from statistics import pstdev
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment import app as deployment_app_module
from tests.predict_probe import build_cases


def _run_cases(case_count: int = 120):
    deployment_app_module.ASSETS = None
    results = []

    with deployment_app_module.app.test_client() as client:
        for case in build_cases(target_count=case_count):
            response = client.post("/predict", json=case.payload)
            assert response.status_code == 200, f"Case {case.case_id} failed: {response.get_data(as_text=True)}"
            payload = response.get_json()
            explainability = payload.get("explainability", {})
            confidence = explainability.get("confidence", {}) if isinstance(explainability, dict) else {}
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "ensemble": float(payload["ensemble_probability"]),
                    "baseline": float(payload["baseline_probability"]),
                    "graphsage": float(payload["graphsage_probability"]),
                    "tgn": float(payload["tgn_probability"]),
                    "risk": payload["risk_classification"],
                    "has_explainability": "explainability" in payload,
                    "top_factor_count": len(explainability.get("top_factors", [])) if isinstance(explainability, dict) else 0,
                    "decision_step_count": len(explainability.get("decision_steps", [])) if isinstance(explainability, dict) else 0,
                    "confidence_score": float(confidence.get("score", 0.0)),
                }
            )

    return results


@pytest.fixture(scope="module")
def results_120_cases():
    return _run_cases(case_count=120)


def test_endpoint_predictions_are_dynamic_across_120_cases(results_120_cases):
    results = results_120_cases
    values = [row["ensemble"] for row in results]

    assert len(results) == 120
    assert len(set(values)) >= 30, "Predictions are still too quantized"
    assert pstdev(values) >= 0.05, "Predictions show insufficient spread"


def test_multiple_risk_classes_present(results_120_cases):
    results = results_120_cases
    counts = Counter(row["risk"] for row in results)

    assert len(counts) >= 2, f"Expected at least 2 risk tiers, got: {dict(counts)}"
    assert any(tier in counts for tier in ["low", "medium", "high", "critical"]), (
        f"Expected at least one elevated tier, got: {dict(counts)}"
    )


def test_key_categories_show_variation(results_120_cases):
    results = results_120_cases

    for category in ["amount", "hour", "type", "unknown", "matrix", "burst"]:
        category_values = [row["ensemble"] for row in results if row["category"] == category]
        assert len(set(category_values)) >= 2, f"Category {category} did not vary"


def test_explainability_payload_is_present_and_populated(results_120_cases):
    results = results_120_cases

    assert all(row["has_explainability"] for row in results), "Explainability object missing in some responses"
    assert sum(1 for row in results if row["top_factor_count"] >= 3) >= 100
    assert sum(1 for row in results if row["decision_step_count"] >= 3) >= 100


def test_confidence_scores_are_valid(results_120_cases):
    results = results_120_cases
    confidence_values = [row["confidence_score"] for row in results]

    assert all(0.0 <= value <= 1.0 for value in confidence_values)
    assert len(set(round(value, 3) for value in confidence_values)) >= 5
