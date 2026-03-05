from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc_and_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float, dict]:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = float(fpr[idx])
    meta = {"threshold": float(thr[idx]), "tpr": float(tpr[idx]), "fpr": float(fpr[idx])}
    return auroc, fpr95, meta


def aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(average_precision_score(y_true, y_score))


def multiclass_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(labels.shape[0]), labels], eps, 1.0)
    return float(-np.log(p).mean())


def multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (float(mask.mean()) * abs(acc_bin - conf_bin))
    return float(ece)


def aurc_from_confidence(confidence: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-confidence)
    sorted_correct = correct[order].astype(np.float32)
    risks = []
    for k in range(1, sorted_correct.shape[0] + 1):
        cov_correct = sorted_correct[:k]
        risks.append(float(1.0 - cov_correct.mean()))
    return float(np.mean(risks))
