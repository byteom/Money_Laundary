from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, float]
    cv_mean_f1: float
    cv_std_f1: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    artifact_dir: str | Path,
) -> dict[str, BaselineResult]:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
    }
    results: dict[str, BaselineResult] = {}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=splitter, scoring="f1")
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = compute_metrics(y_test, predictions, probabilities)
        joblib.dump(model, artifact_root / f"{name}.joblib")
        results[name] = BaselineResult(
            name=name,
            metrics=metrics,
            cv_mean_f1=float(cv_scores.mean()),
            cv_std_f1=float(cv_scores.std()),
        )

    serializable = {
        name: {
            "metrics": result.metrics,
            "cv_mean_f1": result.cv_mean_f1,
            "cv_std_f1": result.cv_std_f1,
        }
        for name, result in results.items()
    }
    (artifact_root / "baseline_metrics.json").write_text(json.dumps(serializable, indent=2))
    return results
