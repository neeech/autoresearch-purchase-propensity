"""Fixed probability metrics; validation only for the baseline search."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, average_precision_score, brier_score_loss, roc_auc_score


def metrics(y, probability):
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0 or p.shape != y.shape:
        raise ValueError('Nonempty aligned one-dimensional predictions required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and between zero and one')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Binary targets required')
    return {'rows': len(y), 'positives': int(y.sum()), 'purchase_rate': float(y.mean()),
            'mean_probability': float(p.mean()),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'brier': float(brier_score_loss(y, p)),
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}


def calibration_table(y, probability, bins=10):
    """Equal-count bins, no learned calibrator; repeated probabilities remain together."""
    d = pd.DataFrame({'target': np.asarray(y), 'probability': np.asarray(probability)})
    if d.probability.nunique() == 1:
        d['bin'] = 0
    else:
        d['bin'] = pd.qcut(d.probability, q=bins, duplicates='drop')
    return d.groupby('bin', observed=True).agg(
        rows=('target', 'size'), positives=('target', 'sum'),
        mean_probability=('probability', 'mean'), observed_rate=('target', 'mean'),
        min_probability=('probability', 'min'), max_probability=('probability', 'max')).reset_index()


def evaluate_segments(y, probability, metadata, train_visitors):
    masks = {'all': np.ones(len(y), dtype=bool),
             'without_prior_day_history': metadata.has_history.eq(0).to_numpy(),
             'with_prior_day_history': metadata.has_history.eq(1).to_numpy(),
             'visitor_unseen_in_train': ~metadata.fullVisitorID.isin(train_visitors).to_numpy(),
             'visitor_seen_in_train': metadata.fullVisitorID.isin(train_visitors).to_numpy()}
    return {name: metrics(np.asarray(y)[mask], np.asarray(probability)[mask])
            for name, mask in masks.items() if mask.any()}
