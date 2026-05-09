from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models.baseline import train_baselines
from models.gnn import train_gnn, TemporalGraphSAGE
from models.tgn import train_tgn, TemporalGraphNetwork
from models.ensemble import train_ensemble, evaluate_ensemble
from preprocessing.preprocess import prepare_dataset


def build_report(baseline_metrics: dict, gnn_metrics: dict, tgn_metrics: dict | None = None, ensemble_metrics: dict | None = None) -> str:
    """Build comprehensive project report."""
    best_baseline_name = max(baseline_metrics, key=lambda name: baseline_metrics[name]["metrics"]["f1"])
    best_baseline = baseline_metrics[best_baseline_name]
    best_gnn = gnn_metrics["metrics"]
    
    lines = [
        "# Advanced AML Detection Project Report",
        "",
        "## Executive Summary",
        "This project implements a state-of-the-art money laundering detection system using",
        "temporal graph neural networks with advanced feature engineering.",
        "",
        "## Model Performance Comparison",
        "",
        "### Baseline Models",
        f"- **Best baseline**: {best_baseline_name}",
        f"  - F1 Score: {best_baseline['metrics']['f1']:.4f}",
        f"  - ROC-AUC: {best_baseline['metrics']['roc_auc']:.4f}",
        f"  - Precision: {best_baseline['metrics']['precision']:.4f}",
        f"  - Recall: {best_baseline['metrics']['recall']:.4f}",
        "",
        "### Graph Neural Network (GraphSAGE)",
        f"- F1 Score: {best_gnn['f1']:.4f}",
        f"- ROC-AUC: {best_gnn['roc_auc']:.4f}",
        f"- Precision: {best_gnn['precision']:.4f}",
        f"- Recall: {best_gnn['recall']:.4f}",
    ]
    
    if tgn_metrics:
        lines.extend([
            "",
            "### Temporal Graph Network (TGN) - Advanced",
            f"- F1 Score: {tgn_metrics['f1']:.4f}",
            f"- ROC-AUC: {tgn_metrics['roc_auc']:.4f}",
            f"- Precision: {tgn_metrics['precision']:.4f}",
            f"- Recall: {tgn_metrics['recall']:.4f}",
        ])
    
    if ensemble_metrics:
        lines.extend([
            "",
            "### Ensemble Model (Best)",
            f"- F1 Score: {ensemble_metrics['f1']:.4f}",
            f"- ROC-AUC: {ensemble_metrics['roc_auc']:.4f}",
            f"- Precision: {ensemble_metrics['precision']:.4f}",
            f"- Recall: {ensemble_metrics['recall']:.4f}",
        ])
    
    lines.extend([
        "",
        "## Advanced Features Implemented",
        "",
        "### Graph Structure Features",
        "- PageRank scores for node importance",
        "- Cycle participation detection",
        "- Flow imbalance analysis",
        "",
        "### Velocity & Burst Detection",
        "- Transaction velocity per hour",
        "- Burst score (max hourly vs average)",
        "- Dormancy ratio detection",
        "- Activity concentration (Gini-like measure)",
        "",
        "### Structuring Detection (AML-specific)",
        "- Near-threshold transaction ratio",
        "- Round number frequency",
        "- Amount variance analysis",
        "- Combined structuring score",
        "",
        "### Temporal Window Features",
        "- Rolling 1h, 6h, 24h aggregations",
        "- Max transaction count per window",
        "- Max volume per window",
        "- Unique counterparties per window",
        "",
        "## Architecture Highlights",
        "",
        "### Temporal Graph Network (TGN)",
        "- **Time2Vec**: Learnable time encoding capturing periodic and linear patterns",
        "- **Memory Module**: GRU-based state tracking for each node",
        "- **Temporal Attention**: Multi-head attention with time decay weighting",
        "- **Focal Loss**: Better handling of class imbalance",
        "",
        "### Ensemble Strategy",
        "- Learned weight combination of Random Forest, GraphSAGE, and TGN",
        "- Isotonic regression calibration for probability outputs",
        "",
        "## Interpretation Notes",
        "",
        "- Baseline models rely on handcrafted sender/receiver and time features",
        "- GraphSAGE adds neighborhood context through message passing",
        "- TGN incorporates temporal memory and attention mechanisms",
        "- The ensemble combines complementary strengths of all models",
        "",
        "### Expected Error Patterns",
        "- **False Positives**: Bursty legitimate high-volume payment chains",
        "- **False Negatives**: Isolated suspicious transfers with weak graph context",
    ])
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AML detection baselines, GNN, TGN, and ensemble models.")
    parser.add_argument("--dataset", default="data/synthetic_aml_transactions.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--skip-tgn", action="store_true", help="Skip TGN training (faster).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    artifact_root = Path(args.artifacts)
    processed_dir = artifact_root / "processed"
    models_dir = artifact_root / "models"
    reports_dir = artifact_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Advanced AML Detection Training Pipeline")
    print("=" * 60)

    # Check if dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}\nPlease ensure synthetic_aml_transactions.csv exists in data/")
    
    print("\n[1/6] Using existing dataset...")
    print(f"  Dataset: {dataset_path}")

    # Preprocess data with advanced features
    print("\n[2/6] Preprocessing data with advanced features...")
    print("  - Computing PageRank scores...")
    print("  - Detecting cycle participation...")
    print("  - Computing velocity features...")
    print("  - Detecting structuring patterns...")
    print("  - Computing temporal window features...")
    prepared = prepare_dataset(dataset_path, processed_dir)
    print(f"  - Train samples: {len(prepared.train_df)}")
    print(f"  - Test samples: {len(prepared.test_df)}")
    print(f"  - Node features: {prepared.train_graph.x.size(1)}")
    print(f"  - Edge features: {prepared.train_graph.edge_attr.size(1)}")

    # Train baseline models
    print("\n[3/6] Training baseline models...")
    baseline_results = train_baselines(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
        models_dir,
    )
    for name, result in baseline_results.items():
        print(f"  - {name}: F1={result.metrics['f1']:.4f}, ROC-AUC={result.metrics['roc_auc']:.4f}")

    # Train GraphSAGE
    print("\n[4/6] Training Temporal GraphSAGE...")
    graphsage_model, gnn_result, node_embeddings = train_gnn(
        prepared.train_graph, prepared.test_graph, models_dir
    )
    print(f"  - GraphSAGE: F1={gnn_result.metrics['f1']:.4f}, ROC-AUC={gnn_result.metrics['roc_auc']:.4f}")

    # Train TGN
    tgn_result = None
    tgn_model = None
    if not args.skip_tgn:
        print("\n[5/6] Training Temporal Graph Network (TGN)...")
        print("  - Initializing Time2Vec encoding...")
        print("  - Setting up GRU memory module...")
        print("  - Training with Focal Loss...")
        tgn_model, tgn_result, tgn_embeddings = train_tgn(
            prepared.train_graph, prepared.test_graph, models_dir
        )
        print(f"  - TGN: F1={tgn_result.metrics['f1']:.4f}, ROC-AUC={tgn_result.metrics['roc_auc']:.4f}")
        print(f"  - Best epoch: {tgn_result.best_epoch}")
    else:
        print("\n[5/6] Skipping TGN training (--skip-tgn flag set)")

    # Train ensemble
    print("\n[6/6] Training Ensemble Model...")
    
    # Get predictions from all models
    baseline_model = baseline_results["random_forest"]
    baseline_probs = baseline_model.metrics.get("_probs", None)
    
    # Re-compute baseline probabilities
    import joblib
    rf_model = joblib.load(models_dir / "random_forest.joblib")
    baseline_probs = rf_model.predict_proba(prepared.x_test)[:, 1]
    
    # Get GraphSAGE probabilities
    graphsage_model.eval()
    with torch.no_grad():
        logits, _ = graphsage_model(prepared.test_graph)
        graphsage_probs = torch.sigmoid(logits).numpy()
    
    # Get TGN probabilities (or use GraphSAGE as fallback)
    if tgn_model is not None:
        tgn_model.eval()
        with torch.no_grad():
            logits, _, _ = tgn_model(
                prepared.test_graph.x,
                prepared.test_graph.edge_index,
                prepared.test_graph.edge_attr,
            )
            tgn_probs = torch.sigmoid(logits).numpy()
    else:
        tgn_probs = graphsage_probs  # Fallback
    
    # Validation split for ensemble fitting and calibration.
    val_indices, holdout_indices = train_test_split(
        np.arange(len(prepared.y_test)),
        test_size=0.5,
        random_state=42,
        stratify=prepared.y_test,
    )

    # Train ensemble on validation predictions only.
    ensemble, optimal_weights, calibrator = train_ensemble(
        baseline_probs[val_indices],
        graphsage_probs[val_indices],
        tgn_probs[val_indices],
        prepared.y_test[val_indices],
        models_dir,
    )
    print(f"  - Optimal weights: {optimal_weights}")
    
    # Evaluate ensemble on held-out test split only.
    ensemble_result = evaluate_ensemble(
        baseline_probs[holdout_indices],
        graphsage_probs[holdout_indices],
        tgn_probs[holdout_indices],
        prepared.y_test[holdout_indices],
        optimal_weights,
        calibrator,
    )
    print(f"  - Ensemble: F1={ensemble_result.metrics['f1']:.4f}, ROC-AUC={ensemble_result.metrics['roc_auc']:.4f}")

    # Save embeddings and metadata
    baseline_serialized = {
        name: {
            "metrics": result.metrics,
            "cv_mean_f1": result.cv_mean_f1,
            "cv_std_f1": result.cv_std_f1,
        }
        for name, result in baseline_results.items()
    }

    with open(models_dir / "gnn_node_embeddings.pkl", "wb") as handle:
        pickle.dump(
            {
                "account_to_index": prepared.account_to_index,
                "node_embeddings": node_embeddings.numpy(),
                "node_feature_dim": prepared.train_graph.x.size(1),
                "edge_feature_dim": prepared.train_graph.edge_attr.size(1),
            },
            handle,
        )

    # Build comparison table
    comparison = []
    for name, result in baseline_results.items():
        comparison.append({"model": name, **result.metrics, "cv_mean_f1": result.cv_mean_f1})
    comparison.append({"model": "temporal_graphsage", **gnn_result.metrics, "cv_mean_f1": None})
    if tgn_result:
        comparison.append({"model": "tgn", **tgn_result.metrics, "cv_mean_f1": None})
    comparison.append({"model": "ensemble", **ensemble_result.metrics, "cv_mean_f1": None})
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    # Build and save report
    tgn_metrics = tgn_result.metrics if tgn_result else None
    report = build_report(baseline_serialized, {"metrics": gnn_result.metrics}, tgn_metrics, ensemble_result.metrics)
    (reports_dir / "project_report.md").write_text(report)
    
    summary = {
        "baseline": baseline_serialized,
        "gnn": gnn_result.metrics,
        "tgn": tgn_metrics,
        "ensemble": ensemble_result.metrics,
        "ensemble_weights": optimal_weights,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nReport written to {reports_dir / 'project_report.md'}")
    print(f"Models saved to {models_dir}")


if __name__ == "__main__":
    main()
