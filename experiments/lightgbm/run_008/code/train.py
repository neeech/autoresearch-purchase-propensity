"""Two baselines: train on train, evaluate validation, leave test reserved."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .evaluate import metrics
from .modeling import build_model


def load_data(directory, split):
    x = pd.read_parquet(directory / f'{split}_X.parquet').drop(columns='month')
    y = pd.read_parquet(directory / f'{split}_y.parquet')
    meta = pd.read_parquet(directory / f'{split}_metadata.parquet')
    if len(x) != len(y) or not y.session_row_id.equals(meta.session_row_id):
        raise ValueError('Partition files do not align')
    forbidden = {'target', 'transactions', 'transactionRevenue', 'fullVisitorID', 'date', 'session_row_id'}
    if forbidden.intersection(x.columns):
        raise ValueError('Outcome or identifiers included in features')
    return x, y.target.to_numpy()


def run(config):
    data = Path(config['data_dir'])
    train_x, train_y = load_data(data, 'train')
    val_x, val_y = load_data(data, 'validation')
    out = Path(config['output_root']) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, out / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    shutil.copyfile(data / 'manifest.json', out / 'data_manifest.json')
    probability = float(train_y.mean())
    model = build_model(train_x.columns.tolist(), config)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train_x, train_y)
    results = pd.DataFrame([
        {'model': 'constant', **metrics(val_y, np.full(len(val_y), probability))},
        {'model': 'logistic', **metrics(val_y, model.predict_proba(val_x)[:, 1])},
    ])
    joblib.dump(model, out / 'model.joblib')
    (out / 'constant.json').write_text(json.dumps({'probability': probability}))
    results.to_csv(out / 'results.csv', index=False)
    (out.parent / 'latest.json').write_text(json.dumps({'run_dir': str(out)}, indent=2))
    report = Path('reports')
    report.mkdir(exist_ok=True)
    intro = 'Entrenamiento: enero–abril. Validación: mayo. Test reservado. Logística con contexto, actividad e historia; imputación y codificación ajustadas sólo en train. Sin calibración ni búsqueda de configuraciones.'
    rows = '\n'.join(f'| {r.model} | {r.log_loss:.5f} | {r.average_precision:.4f} |' for r in results.itertuples())
    (report / 'baselines.md').write_text('# Baselines simples\n\n' + intro + '\n\n| Modelo | Log loss ↓ | Average Precision ↑ |\n|---|---:|---:|\n' + rows + f'\n\nArtefactos: `{out}`.\n')
    (report / 'baselines.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>Baselines simples</title><body><h1>Baselines simples</h1><p>' + intro + '</p>' + results.to_html(index=False, float_format=lambda v: f'{v:.5f}') + '</body></html>')
    print(results.to_string(index=False))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baselines.json')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
