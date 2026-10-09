"""Small experiment loop: validated JSON, subprocess training and JSONL history."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError

import numpy as np
from sklearn.metrics import brier_score_loss

from .evaluate import metrics
from .train import load_data
from .research_agent import ResearchAgent

ROOT = Path(__file__).resolve().parents[2]
FEATURES, MODEL = 'configs/autoresearch_features.json', 'configs/autoresearch_model.json'
ALLOWED = (FEATURES, MODEL)
PARAMS = {
    'logistic': ['C', 'solver', 'max_iter', 'tol', 'penalty'],
    'random_forest': ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'bootstrap'],
    'extra_trees': ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'bootstrap'],
    'lightgbm': ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples',
                 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample', 'subsample_freq'],
}


def validate(code, proposal):
    """Return candidate configs; never apply LLM changes to the repository."""
    if not isinstance(proposal, dict) or set(proposal) != {'hypothesis', 'reasoning', 'changes'}:
        raise ValueError('Expected hypothesis, reasoning and changes')
    if not all(isinstance(proposal[k], str) and proposal[k].strip() for k in ('hypothesis', 'reasoning')):
        raise ValueError('Hypothesis and reasoning required')
    if not isinstance(proposal['changes'], list) or not proposal['changes']:
        raise ValueError('Changes required')
    candidate = code.copy()
    for change in proposal['changes']:
        if not isinstance(change, dict) or set(change) != {'file', 'old', 'new'}:
            raise ValueError('Expected file, old and new')
        name, old, new = change['file'], change['old'], change['new']
        if name not in ALLOWED or not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError('Only literal replacements in allowed JSON files')
        if candidate[name].count(old) != 1 or len(new) > 16384:
            raise ValueError('Replacement must match exactly once; maximum 16 KB')
        candidate[name] = candidate[name].replace(old, new, 1)
    configs = {name: json.loads(text) for name, text in candidate.items()}
    model, features = configs[MODEL], configs[FEATURES]
    if (set(model) != {'model', 'params', 'min_category_frequency'} or model['model'] not in PARAMS
            or not isinstance(model['params'], dict) or set(model['params']) - set(PARAMS[model['model']])):
        raise ValueError('Unsupported model or hyperparameter')
    for value in model['params'].values():
        if type(value) not in (str, int, float, bool, type(None)) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Parameters must be finite scalar JSON values')
    for name in ('n_estimators', 'max_iter'):
        value = model['params'].get(name, 100)
        if type(value) is not int or not 1 <= value <= 500:
            raise ValueError('Maximum 500 trees/iterations')
    if type(model['min_category_frequency']) is not int or model['min_category_frequency'] < 1:
        raise ValueError('Invalid category frequency')
    if (set(features) != {'add', 'drop'} or not isinstance(features['add'], list)
            or not isinstance(features['drop'], list) or len(features['add']) > 32):
        raise ValueError('Expected add and drop lists; maximum 32 new features')
    for feature in features['add']:
        if not isinstance(feature, dict) or set(feature) != {'name', 'op', 'columns'}:
            raise ValueError('Expected feature name, op and columns')
        count = 1 if feature['op'] == 'log1p' else 2
        if feature['op'] not in {'log1p', 'ratio', 'product', 'sum', 'difference'}:
            raise ValueError('Unsupported feature operation')
        if not isinstance(feature['columns'], list) or len(feature['columns']) != count:
            raise ValueError('Incorrect number of columns')
        if not all(isinstance(c, str) and c.isidentifier() for c in [feature['name'], *feature['columns']]):
            raise ValueError('Invalid feature name')
    if not all(isinstance(c, str) and c.isidentifier() for c in features['drop']):
        raise ValueError('Invalid dropped column')
    if candidate == code:
        raise ValueError('Proposal has no changes')
    return candidate


def save_configs(code, directory):
    directory.mkdir(exist_ok=True)
    for name, text in code.items():
        (directory / Path(name).name).write_text(text)


def is_better(score, best, maximize=False):
    return score > best if maximize else score < best


def experiment(code, data, directory, source, timeout):
    """The subprocess executes only fixed project code; LLM input is JSON data."""
    directory.mkdir()
    save_configs(code, directory)
    env = {**os.environ, 'PYTHONPATH': str(source), 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4'}
    env.pop('GROQ_API_KEY', None)
    command = [sys.executable, '-m', 'autoresearch_purchase_propensity.research_train', str(data), str(directory)]
    with (directory / 'training.log').open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=directory, stdout=log, stderr=log, start_new_session=True)
        try:
            if process.wait(timeout=timeout):
                raise RuntimeError(f'Training failed; see {directory / "training.log"}')
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return np.load(directory / 'predictions.npy', allow_pickle=False)


def record(output, history, **entry):
    entry['timestamp'] = datetime.now(timezone.utc).isoformat()
    with (output / 'history.jsonl').open('a') as file:
        file.write(json.dumps(entry, allow_nan=False) + '\n')
    history.append(entry)


def run(agent, code, data, output, source, n_experiments=30, timeout=180):
    x, y = load_data(data, 'validation')
    history, best, initial = [], None, None
    for number in range(n_experiments + 1):
        started, before = time.monotonic(), best
        stop = False
        proposal, score, values, error = {}, None, {}, None
        status = 'failed'
        try:
            candidate = code
            if number:
                proposal = agent.propose({'goal': 'Minimize May validation log loss', 'best_score': best,
                    'code': code, 'available_columns': x.columns.tolist(), 'model_parameters': PARAMS,
                    'history': history[-3:], 'tested_hypotheses': [e['hypothesis'] for e in history]})
                status = 'invalid'
                candidate = validate(code, proposal)
            status = 'failed'
            predictions = experiment(candidate, data, output / f'run_{number:03d}', source, timeout)
            values = metrics(y, predictions)
            values['brier_score'] = float(brier_score_loss(y, predictions))
            score = values['log_loss']
            status = 'baseline' if not number else 'keep' if is_better(score, best) else 'discard'
            if status in ('baseline', 'keep'):
                code, best = candidate, score
                save_configs(code, output / 'best')
            if not number:
                initial = score
        except Exception as exception:
            error = str(exception)
            stop = isinstance(exception, HTTPError)
            if isinstance(exception, subprocess.TimeoutExpired):
                status = 'timeout'
        proposal = proposal if isinstance(proposal, dict) else {}
        record(output, history, experiment_id=number, hypothesis=proposal.get('hypothesis', 'Baseline'),
               reasoning=proposal.get('reasoning', ''), changes=proposal.get('changes', []),
               score_before=before, score_after=score, status=status, metrics=values,
               duration=time.monotonic() - started, error=error)
        print(f'Experiment {number}: {status.upper()} | log loss: {score}\n'
              f'{proposal.get("hypothesis", "Baseline")}' + (f'\n{error}' if error else ''), flush=True)
        if best is None or stop:
            break  # Stop on missing baseline, authentication errors or API quota exhaustion.
    kept = [e['experiment_id'] for e in history if e['status'] in ('baseline', 'keep')]
    summary = {'initial_score': initial, 'best_score': best, 'total_experiments': len(history) - 1,
               'best_experiment_id': kept[-1] if kept else None}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('AutoResearch finished\n' + json.dumps(summary, indent=2), flush=True)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--model', default='openai/gpt-oss-120b')
    parser.add_argument('--check-api', action='store_true', help='Check credentials and model without training')
    args = parser.parse_args()
    if not 1 <= args.experiments <= 30 or not 1 <= args.timeout <= 180:
        parser.error('Maximum 30 experiments and 180 seconds per experiment')
    try:
        agent = ResearchAgent(args.model)
        agent.check()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.check_api:
        print(f'Groq connection OK. Model: {args.model}')
        return
    output = ROOT / 'experiments/autoresearch' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    data = output / 'data'
    data.mkdir()
    for split in ('train', 'validation'):
        for suffix in ('X', 'y', 'metadata'):
            name = f'{split}_{suffix}.parquet'
            shutil.copyfile(ROOT / 'data/processed' / name, data / name)
    source = output / 'code'
    package = source / 'autoresearch_purchase_propensity'
    package.mkdir(parents=True)
    for name in ('__init__', 'train', 'modeling', 'evaluate', 'research_features', 'research_train'):
        shutil.copyfile(Path(__file__).parent / f'{name}.py', package / f'{name}.py')
    (output / 'protocol.json').write_text(json.dumps(vars(args), indent=2))
    run(agent, {name: (ROOT / name).read_text() for name in ALLOWED}, data, output, source,
        args.experiments, args.timeout)
