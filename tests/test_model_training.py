"""Unit tests for anomaly detection model pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
import numpy as np
import pytest


def test_data_generation():
    from train_anomaly_model import generate_synthetic_data
    X, y = generate_synthetic_data(n_normal=100, n_anomaly=20)
    assert X.shape == (120, 64, 1)
    assert set(y).issubset({0.0, 1.0})


def test_model_builds():
    from train_anomaly_model import build_model
    model = build_model(seq_len=64)
    assert model.output_shape == (None, 1)


def test_model_predicts_shape():
    from train_anomaly_model import build_model
    model = build_model()
    x = np.random.randn(4, 64, 1).astype(np.float32)
    pred = model.predict(x, verbose=0)
    assert pred.shape == (4, 1)
    assert all(0 <= p <= 1 for p in pred.flatten())
