"""Evaluate frozen models on June; never train or select thresholds here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .evaluate import metrics
from .marketing import compare


def verify_files(root, expected):
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input changed: {path}')


def evaluate(config, output):
    data, output = Path(config['data_dir']).resolve(), Path(output).resolve()
    verify_files(data, config['data_sha256'])
    for model in config['models']:
        verify_files(Path(model['artifact']), model['artifact_sha256'])
        if set(model['thresholds']) != set(config['penalties']):
            raise ValueError('Each model needs frozen thresholds for every scenario')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(config, indent=2))
    rows, costs, segments = [], [], []
    reference = None
    metadata = pd.read_parquet(data / 'test_metadata.parquet')
    for model in config['models']:
        directory = output / model['name']
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/export_predictions.py'),
                        model['artifact'], '--split', 'test', '--data', str(data),
                        '--output', str(directory)], check=True)
        frame = pd.read_csv(directory / ('constant.csv' if model['kind'] == 'constant' else 'candidate.csv'))
        identity = frame[['session_row_id', 'target']]
        if reference is not None and not reference.equals(identity):
            raise ValueError('Models evaluated on different rows/labels')
        reference = identity
        if not frame.session_row_id.equals(metadata.session_row_id):
            raise ValueError('Prediction and metadata rows differ')
        rows.append({'model': model['name'], **metrics(frame.target, frame.probability)})
        report, thresholds = compare(frame, config['penalties'], model['thresholds'], config['contact_cost'])
        report.insert(0, 'model', model['name'])
        costs.append(report)
        (directory / 'thresholds.json').write_text(json.dumps(thresholds, indent=2))
        for present in (False, True):
            group = frame.loc[metadata.has_history.to_numpy().astype(bool) == present]
            if len(group):
                segments.append({'model': model['name'], 'has_history': present,
                                 'rows': len(group), **metrics(group.target, group.probability)})
    result = pd.DataFrame(rows)
    result.to_csv(output / 'metrics.csv', index=False)
    pd.concat(costs, ignore_index=True).to_csv(output / 'costs.csv', index=False)
    pd.DataFrame(segments).to_csv(output / 'segments.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/test_evaluation.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mlflow', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    results = evaluate(config, args.output)
    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('purchase-propensity-test')
        for row in results.to_dict('records'):
            with mlflow.start_run(run_name=row['model'] + '-test-2017-06'):
                mlflow.set_tags({'split': 'test', 'thresholds': 'frozen_from_may', 'approval': 'pending'})
                mlflow.log_metrics({k: v for k, v in row.items() if k != 'model' and pd.notna(v)})
                for name in ('protocol.json', 'metrics.csv', 'costs.csv', 'segments.csv'):
                    mlflow.log_artifact(str(args.output / name))
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
