"""Retrospective marketing penalties; never an estimate of campaign uplift."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import metrics


def threshold_curve(y, p, costs):
    """All distinct decisions for p >= threshold; ties favor fewer contacts."""
    metrics(y, p)  # Reuse the protected probability/target checks.
    if set(costs) != {'tp', 'tn', 'fp', 'fn'} or any(
            not np.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError('Four finite nonnegative penalties required')
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    order = np.argsort(-p, kind='stable')
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]) + 1, len(p)]
    contacts = np.r_[0, ends]
    tp = np.r_[0, np.cumsum(y)[ends - 1]]
    fp, fn = contacts - tp, y.sum() - tp
    tn = len(y) - y.sum() - fp
    return pd.DataFrame({
        'threshold': np.r_[np.nextafter(1., np.inf), p[ends - 1]],
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'contact_rate': contacts / len(y),
        'precision': np.divide(tp, contacts, out=np.zeros(len(tp)), where=contacts > 0),
        'recall': tp / max(1, y.sum()),
        'mean_cost': (tp*costs['tp'] + tn*costs['tn'] + fp*costs['fp'] + fn*costs['fn']) / len(y),
    })


def compare(frame, scenarios, frozen=None, contact_cost=1.):
    if not np.isfinite(contact_cost) or contact_cost < 0:
        raise ValueError('Contact cost must be finite and nonnegative')
    required = {'session_row_id', 'target', 'probability'}
    if not required.issubset(frame) or frame.session_row_id.isna().any() or frame.session_row_id.duplicated().any():
        raise ValueError('Unique session_row_id, target and probability required')
    rows, thresholds = [], {}
    for name, costs in scenarios.items():
        curve = threshold_curve(frame.target, frame.probability, costs)
        threshold = (float(frozen[name]) if frozen is not None
                     else float(curve.loc[curve.mean_cost.idxmin(), 'threshold']))
        if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf):
            raise ValueError('Invalid frozen threshold')
        thresholds[name] = threshold
        # Locate the equivalent decision in the curve, without selecting on evaluation labels.
        contacts = int((frame.probability >= threshold).sum())
        decision = curve.iloc[np.flatnonzero((curve.tp + curve.fp).to_numpy() == contacts)[0]].copy()
        decision['threshold'] = threshold
        for strategy, row in [('model', decision), ('contact_none', curve.iloc[0]), ('contact_all', curve.iloc[-1])]:
            rows.append({'scenario': name, 'strategy': strategy, **row.to_dict()})
    report = pd.DataFrame(rows)
    report['contacts'] = (report.tp + report.fp).astype(int)
    report['direct_spend'] = report.contacts * contact_cost
    report['total_penalty'] = report.mean_cost * len(frame)
    return report, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path, help='CSV: session_row_id,target,probability')
    parser.add_argument('--costs', type=Path, default=Path('configs/marketing.json'))
    parser.add_argument('--thresholds', type=Path, help='Frozen validation thresholds for a separate period')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-id', required=True, help='Immutable training run/commit identifier')
    parser.add_argument('--period', required=True, help='Descriptive label, e.g. validation-2017-05')
    parser.add_argument('--contact-cost', type=float, default=1., help='Simulated direct cost per contact; distinct from penalties')
    parser.add_argument('--mlflow', action='store_true')
    args = parser.parse_args()
    costs = json.loads(args.costs.read_text())
    frozen = json.loads(args.thresholds.read_text()) if args.thresholds else None
    frame = pd.read_csv(args.predictions)
    report, thresholds = compare(frame, costs, frozen, args.contact_cost)
    args.output.mkdir(parents=True, exist_ok=False)
    report.to_csv(args.output / 'costs.csv', index=False)
    (args.output / 'thresholds.json').write_text(json.dumps(thresholds, indent=2))
    metadata = {'model_id': args.model_id, 'period': args.period,
                'mode': 'frozen_evaluation' if frozen is not None else 'threshold_selection',
                'simulation_only': True, 'contact_cost': args.contact_cost, 'costs': costs, 'metrics': metrics(frame.target, frame.probability),
                'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    (args.output / 'evaluation.json').write_text(json.dumps(metadata, indent=2))
    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('purchase-propensity-marketing')
        with mlflow.start_run(run_name=f'{args.model_id}-{args.period}'):
            mlflow.set_tags({'simulation_only': 'true', 'approval': 'not_approved'})
            mlflow.log_params({k: metadata[k] for k in ('model_id', 'period', 'mode', 'predictions_sha256')})
            mlflow.log_metrics({k: v for k, v in metadata['metrics'].items() if v is not None})
            for row in report.to_dict('records'):
                for metric in ('mean_cost', 'direct_spend', 'total_penalty', 'contacts'):
                    mlflow.log_metric(f"{row['scenario']}.{row['strategy']}.{metric}", row[metric])
            mlflow.log_artifacts(str(args.output), artifact_path='evaluation')
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
