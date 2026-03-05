"""Smoke checks for metric helpers used in OOD evaluation."""

import numpy as np

from src.metrics.ood_metrics import (
    aupr,
    aurc_from_confidence,
    auroc_and_fpr95,
    multiclass_ece,
    multiclass_nll,
)


def test_ood_metric_smoke() -> None:
    id_scores = np.array([0.1, 0.2, 0.3])
    ood_scores = np.array([0.7, 0.8, 0.9])
    auroc, fpr95, _ = auroc_and_fpr95(id_scores, ood_scores)
    ap = aupr(id_scores, ood_scores)
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= fpr95 <= 1.0
    assert 0.0 <= ap <= 1.0


def test_calibration_metric_smoke() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=np.float64)
    labels = np.array([0, 1, 1], dtype=np.int64)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float32)
    nll = multiclass_nll(probs, labels)
    ece = multiclass_ece(probs, labels, n_bins=5)
    aurc = aurc_from_confidence(conf, correct)
    assert nll >= 0.0
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= aurc <= 1.0
