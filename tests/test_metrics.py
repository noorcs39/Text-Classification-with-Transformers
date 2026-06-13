from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_classification.metrics import compute_metrics


def test_compute_metrics_returns_expected_keys():
    logits = np.array([[2.0, 0.5], [0.1, 3.0], [1.0, 0.2]])
    labels = np.array([0, 1, 0])

    metrics = compute_metrics((logits, labels))

    assert set(metrics) == {"accuracy", "f1", "precision", "recall"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] == 1.0
