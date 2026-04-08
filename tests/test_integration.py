"""
Integration tests for the full AML detection pipeline.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import torch


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_advanced_feature_computation(self):
        """Test that advanced features are computed correctly."""
        # Create minimal test data
        data = pd.DataFrame({
            "transaction_id": [f"T{i}" for i in range(20)],
            "sender_id": ["ACC_001"] * 10 + ["ACC_002"] * 10,
            "receiver_id": ["ACC_002"] * 10 + ["ACC_003"] * 10,
            "transaction_amount": [1000.0] * 5 + [9500.0] * 5 + [500.0] * 10,  # Some near-threshold
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="h"),  # lowercase h for newer pandas
            "transaction_type": ["transfer"] * 20,
            "label": [0] * 15 + [1] * 5,
        })
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_data.csv"
            data.to_csv(data_path, index=False)
            
            # Import here to avoid issues if module not ready
            try:
                from preprocessing.preprocess import _build_node_features, _compute_structuring_features
                
                accounts = ["ACC_001", "ACC_002", "ACC_003"]
                node_frame = _build_node_features(data, accounts)
                
                # Check that advanced features exist
                assert "pagerank" in node_frame.columns
                assert "burst_score" in node_frame.columns
                assert "structuring_score" in node_frame.columns
                assert "tx_count_1h_max" in node_frame.columns
                
                # Check structuring detection
                struct_features = _compute_structuring_features(data, accounts)
                acc1_struct = struct_features.get("ACC_001", {})
                # ACC_001 has some near-threshold transactions (9500 near 10000)
                assert acc1_struct.get("near_threshold_ratio", 0) > 0
                
            except ImportError:
                pytest.skip("Preprocessing module not available")


class TestModels:
    """Tests for model modules."""
    
    def test_tgn_components_exist(self):
        """Verify TGN components are importable."""
        from models.tgn import (
            Time2Vec,
            TemporalAttention,
            MemoryModule,
            MessageAggregator,
            FocalLoss,
            TemporalGraphNetwork,
        )
        
        # All should be classes
        assert callable(Time2Vec)
        assert callable(TemporalAttention)
        assert callable(MemoryModule)
        assert callable(MessageAggregator)
        assert callable(FocalLoss)
        assert callable(TemporalGraphNetwork)
    
    def test_ensemble_components_exist(self):
        """Verify ensemble components are importable."""
        from models.ensemble import (
            LearnedWeightEnsemble,
            EnsemblePredictor,
            train_ensemble,
            evaluate_ensemble,
        )
        
        assert callable(LearnedWeightEnsemble)
        assert callable(EnsemblePredictor)
        assert callable(train_ensemble)
        assert callable(evaluate_ensemble)
    
    def test_tgn_forward_shape(self):
        """Test TGN forward pass produces correct shapes."""
        from models.tgn import TemporalGraphNetwork
        
        model = TemporalGraphNetwork(
            node_dim=32,
            edge_dim=9,
            memory_dim=64,
            time_dim=16,
            hidden_dim=64,
        )
        
        num_nodes = 20
        num_edges = 50
        
        node_features = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 9)
        
        logits, memory, _ = model(node_features, edge_index, edge_attr)
        
        assert logits.shape == (num_edges,)
        assert memory.shape == (num_nodes, 64)


class TestEnsemble:
    """Tests for ensemble model."""
    
    def test_learned_weight_ensemble(self):
        """Test learned weight ensemble combination."""
        from models.ensemble import LearnedWeightEnsemble
        
        ensemble = LearnedWeightEnsemble(num_models=3)
        
        # Fake predictions from 3 models
        predictions = torch.rand(10, 3)
        
        combined = ensemble(predictions)
        
        assert combined.shape == (10,)
        assert (combined >= 0).all() and (combined <= 1).all()
    
    def test_ensemble_evaluation(self):
        """Test ensemble evaluation computes metrics."""
        from models.ensemble import evaluate_ensemble
        
        # Fake predictions
        np.random.seed(42)
        baseline_probs = np.random.rand(100)
        graphsage_probs = np.random.rand(100)
        tgn_probs = np.random.rand(100)
        y_true = np.random.randint(0, 2, 100)
        
        weights = {"baseline": 0.2, "graphsage": 0.3, "tgn": 0.5}
        
        result = evaluate_ensemble(
            baseline_probs, graphsage_probs, tgn_probs,
            y_true, weights, calibrator=None
        )
        
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert "roc_auc" in result.metrics
        assert result.model_weights == weights


class TestEndToEnd:
    """End-to-end pipeline tests."""
    
    def test_graph_batch_creation(self):
        """Test graph batch can be created from preprocessed data."""
        from graph.temporal_graph import GraphBatch
        
        batch = GraphBatch(
            x=torch.randn(10, 32),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 9),
            labels=torch.randint(0, 2, (20,)).float(),
            edge_ids=torch.arange(20),
        )
        
        assert batch.x.shape[0] == 10
        assert batch.edge_index.shape[1] == 20
        assert batch.labels.shape[0] == 20
    
    def test_model_training_loop(self):
        """Test that a training loop completes without error."""
        from models.tgn import TemporalGraphNetwork, FocalLoss
        from graph.temporal_graph import GraphBatch
        
        # Create fake data
        batch = GraphBatch(
            x=torch.randn(20, 32),
            edge_index=torch.randint(0, 20, (2, 50)),
            edge_attr=torch.randn(50, 9),
            labels=torch.randint(0, 2, (50,)).float(),
            edge_ids=torch.arange(50),
        )
        
        model = TemporalGraphNetwork(32, 9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = FocalLoss()
        
        # Run a few training steps
        for _ in range(3):
            model.train()
            optimizer.zero_grad()
            
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Should complete without error
        assert True


class TestDeployment:
    """Tests for deployment module."""
    
    def test_flask_app_exists(self):
        """Verify Flask app can be imported."""
        from deployment.app import app, health
        
        assert app is not None
        
        # Test health endpoint
        with app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
