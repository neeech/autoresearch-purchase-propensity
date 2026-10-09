"""Freeze selected artifacts, then evaluate June once without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from .modeling import CATEGORIES

ROOT = Path('experiments/final_test')
LOCK = Path('configs/final_evaluation.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    if LOCK.exists():
        raise ValueError('Final evaluation is already frozen')
    state = json.loads(Path('research-state.json').read_text())
    best = Path(state['best']['experiment'])
    baseline = Path(json.loads(Path('experiments/baselines/latest.json').read_text())['run_dir'])
    paths = {'lightgbm': str(best / 'model.txt'), 'logistic': str(baseline / 'model.joblib')}
    locked = {'models': paths, 'sha256': {name: digest(path) for name, path in paths.items()},
              'selected_config': json.loads((best / 'config.json').read_text()),
              'selection_metrics': state['best'], 'test_period': '2017-06',
              'policy': 'No refitting, tuning or calibration; one evaluation of June.',
              'evaluation_code_sha256': digest(__file__)}
    LOCK.write_text(json.dumps(locked, indent=2) + '\n')


def evaluate():
    locked = json.loads(LOCK.read_text())
    assert digest(__file__) == locked['evaluation_code_sha256']
    for name, path in locked['models'].items():
        assert digest(path) == locked['sha256'][name], 'Frozen model changed'
    ROOT.mkdir(parents=True, exist_ok=False)  # Prevent accidental repeated evaluation.
    directory = Path(locked['selected_config']['data_dir'])
    x = pd.read_parquet(directory / 'test_X.parquet').drop(columns='month')
    y = pd.read_parquet(directory / 'test_y.parquet')
    meta = pd.read_parquet(directory / 'test_metadata.parquet')
    assert len(x) == len(y) == len(meta) and y.session_row_id.equals(meta.session_row_id)
    assert meta.date.dt.strftime('%Y-%m').eq(locked['test_period']).all()
    logistic = joblib.load(locked['models']['logistic'])
    probabilities = {'logistic': logistic.predict_proba(x)[:, 1]}
    booster = lgb.Booster(model_file=locked['models']['lightgbm'])
    assert booster.feature_name() == x.columns.tolist()
    for column in set(CATEGORIES) & set(x.columns):
        x[column] = x[column].astype('string').astype('category')
    probabilities['lightgbm'] = booster.predict(x, num_threads=4)
    rows = []
    for model, p in probabilities.items():
        for segment, mask in {'all': meta.has_history.notna(), 'without_history': meta.has_history.eq(0),
                              'with_history': meta.has_history.eq(1)}.items():
            target = y.target[mask]
            rows.append({'model': model, 'segment': segment, 'rows': int(mask.sum()),
                         'positives': int(target.sum()), 'log_loss': float(log_loss(target, p[mask], labels=[0, 1])),
                         'brier': float(brier_score_loss(target, p[mask]))})
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'metrics.csv', index=False)
    pd.DataFrame({'session_row_id': y.session_row_id, 'target': y.target, **probabilities}).to_parquet(ROOT / 'predictions.parquet', index=False)
    hashes = {path.name: digest(path) for path in directory.glob('test_*.parquet')}
    (ROOT / 'provenance.json').write_text(json.dumps({'test_sha256': hashes, 'lock_sha256': digest(LOCK)}, indent=2))
    print(results.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else evaluate()
