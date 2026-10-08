"""Baseline metrics: lower log loss, higher Average Precision."""
import numpy as np
from sklearn.metrics import log_loss, average_precision_score


def metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    if not len(y) or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('Aligned, nonempty, finite predictions required')
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Binary targets and probabilities between 0 and 1 required')
    return {'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None}
