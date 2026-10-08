"""Bounded experiments proposed by the conversational agent; never reads test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from .train import load_data
from .modeling import CATEGORIES

STATE = Path('research-state.json')
CONFIG = Path('configs/lightgbm.json')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(folder):
    config = json.loads((folder / 'config.json').read_text())
    train, y = load_data(Path(config['data_dir']), 'train')
    val, target = load_data(Path(config['data_dir']), 'validation')
    for column in set(CATEGORIES) & set(train.columns):
        categories = train[column].dropna().astype(str).unique().tolist()
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype('string').astype(dtype)
        val[column] = val[column].astype('string').astype(dtype)
    params = dict(config['params'], objective='binary', verbosity=-1,
                  n_jobs=config['threads'], random_state=config['seed'],
                  deterministic=True, force_col_wise=True)
    started = time.time()
    model = lgb.LGBMClassifier(**params).fit(train, y)  # No eval_set or early stopping.
    if model.n_estimators_ != config['params']['n_estimators']:
        raise RuntimeError('Training did not complete the configured number of trees')
    p = model.predict_proba(val)[:, 1]
    model.booster_.save_model(str(folder / 'model.txt'))
    pd.DataFrame({'target': target, 'probability': p}).to_parquet(folder / 'predictions.parquet', index=False)
    save(folder / 'result.json', {'status': 'ok', 'log_loss': float(log_loss(target, p)),
                                 'brier': float(brier_score_loss(target, p)),
                                 'seconds': time.time() - started, 'trees': model.n_estimators_})


def prepare():
    config = json.loads(CONFIG.read_text())
    if not STATE.exists():
        save(STATE, {'started': time.time(), 'max_experiments': config['max_experiments'],
                     'budget_minutes': config['budget_minutes'], 'completed': 0, 'best': None})
    state = json.loads(STATE.read_text())
    if state.get('pending'):
        raise ValueError('Run the pending experiment before proposing another')
    if state['completed'] >= state['max_experiments'] or time.time() >= state['started'] + state['budget_minutes'] * 60:
        raise ValueError('Research budget exhausted')
    folder = Path(config['output_dir']) / f"run_{state['completed'] + 1:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / 'config.json', config)
    shutil.copytree(Path(__file__).parent, folder / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    manifest = Path(config['data_dir']) / 'manifest.json'
    shutil.copyfile(manifest, folder / 'data_manifest.json')
    import numpy, sklearn
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for split in ['train', 'validation'] for p in Path(config['data_dir']).glob(f'{split}_*.parquet')}
    save(folder / 'provenance.json', {'inputs': inputs, 'features': json.loads(manifest.read_text())['features'],
                                    'versions': {'lightgbm': lgb.__version__, 'pandas': pd.__version__,
                                                 'numpy': numpy.__version__, 'sklearn': sklearn.__version__, 'python': sys.version}})
    (folder / 'protocol.md').write_text(f"# {folder.name}\n\n{config['hypothesis']}\n\nObjetivo: reducir log loss en mayo; Brier secundario. Features fijas, CPU, sin early stopping. Junio reservado.\n")
    state['pending'] = str(folder)
    save(STATE, state)
    print(folder)


def run():
    state = json.loads(STATE.read_text())
    folder = Path(state['pending'])
    # Require the exact protocol/config/code snapshot to be committed before training.
    for path in [folder / 'protocol.md', folder / 'config.json', *list((folder / 'code').glob('*.py'))]:
        committed = subprocess.check_output(['git', 'show', f'HEAD:{path}'])
        if committed != path.read_bytes():
            raise ValueError(f'Commit the protocol before training: {path}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remaining = state['started'] + state['budget_minutes'] * 60 - time.time()
    if remaining <= 0:
        raise ValueError('Research time budget exhausted')
    with (folder / 'training.log').open('w') as log:
        try:
            process = subprocess.run([sys.executable, '-m', __spec__.name, '--worker', str(folder)],
                                     stdout=log, stderr=log, timeout=remaining)
            if process.returncode:
                raise RuntimeError(f'Training failed; inspect {folder}/training.log')
            result = json.loads((folder / 'result.json').read_text())
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            result = {'status': 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'error', 'error': str(error)}
    better = result['status'] == 'ok' and (state['best'] is None or result['log_loss'] < state['best']['log_loss'])
    result.update(decision='keep' if better else 'discard', protocol_commit=commit)
    save(folder / 'result.json', result)
    if better:
        state['best'] = {'experiment': str(folder), 'log_loss': result['log_loss'], 'brier': result['brier']}
    state['completed'] += 1
    state.pop('pending')
    save(STATE, state)
    rows = []
    for path in sorted(folder.parent.glob('run_*/result.json')):
        rows.append({'experiment': path.parent.name, **json.loads(path.read_text())})
    pd.DataFrame(rows).to_csv(folder.parent / 'history.csv', index=False)
    with Path('research-log.md').open('a') as log:
        log.write(f"- {folder.name}: {result}\n")
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        parser.error('Choose --prepare or --run')
