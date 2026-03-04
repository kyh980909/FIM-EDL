from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc_and_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float, dict]:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = float(fpr[idx])
    meta = {"threshold": float(thr[idx]), "tpr": float(tpr[idx]), "fpr": float(fpr[idx])}
    return auroc, fpr95, meta
