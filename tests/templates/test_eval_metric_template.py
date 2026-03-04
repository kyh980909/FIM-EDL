"""Template: replace with plugin-specific metric validation case."""

import numpy as np

from src.metrics.ood_metrics import auroc_and_fpr95


def test_ood_metric_smoke() -> None:
    id_scores = np.array([0.1, 0.2, 0.3])
    ood_scores = np.array([0.7, 0.8, 0.9])
    auroc, fpr95, _ = auroc_and_fpr95(id_scores, ood_scores)
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= fpr95 <= 1.0
