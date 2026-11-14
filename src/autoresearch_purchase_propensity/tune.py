"""LightGBM + Optuna reference, using the existing temporal training pipeline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import optuna
import pandas as pd

from .train import run


def tune(config, output, trials=20, tracking=False):
    if trials < 1:
        raise ValueError('At least one trial required')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(config['data_dir']).resolve()
    frozen = output / 'data'
    frozen.mkdir()
    inputs = [source / f'{split}_{suffix}.parquet'
              for split in ('train', 'validation') for suffix in ('X', 'y', 'metadata')]
    inputs.append(source / 'manifest.json')
    hashes = {}
    for path in inputs:
        shutil.copyfile(path, frozen / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fixed = {**config, 'model': 'lightgbm', 'data_dir': str(frozen),
             'output_root': str(output / 'training')}
    (output / 'protocol.json').write_text(json.dumps({
        'config': fixed, 'trials': trials, 'sha256': hashes, 'metric': 'validation log_loss',
        'early_stopping': False, 'seed': config['seed'], 'simulation_only': True,
    }, indent=2))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config['seed']),
                                storage='sqlite:///' + str(output / 'optuna.db'), study_name='lightgbm')

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 350, step=50),
            'learning_rate': trial.suggest_float('learning_rate', .02, .15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 48),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 400),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .7, 1.),
            'reg_lambda': trial.suggest_float('reg_lambda', .01, 10., log=True),
        }
        started = time.monotonic()
        artifact = run({**fixed, 'lightgbm': params})
        result = pd.read_csv(artifact / 'results.csv').set_index('model').loc['candidate']
        trial.set_user_attr('artifact', str(artifact))
        trial.set_user_attr('duration_seconds', time.monotonic() - started)
        if tracking:
            import mlflow
            with mlflow.start_run(run_name=f'lightgbm-trial-{trial.number}'):
                mlflow.log_params(params)
                mlflow.set_tags({'artifact': str(artifact), 'approval': 'not_approved', 'split': 'validation'})
                mlflow.log_metrics(result.to_dict())
                mlflow.log_metric('duration_seconds', trial.user_attrs['duration_seconds'])
                for name in ('config.json', 'data_manifest.json', 'results.csv'):
                    mlflow.log_artifact(str(artifact / name))
        return float(result.log_loss)

    if tracking:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('purchase-propensity-optuna')
    study.optimize(objective, n_trials=trials, n_jobs=1)
    study.trials_dataframe().to_csv(output / 'trials.csv', index=False)
    best = study.best_trial
    best_config = {**fixed, 'lightgbm': best.params}
    (output / 'best_config.json').write_text(json.dumps(best_config, indent=2))
    summary = {'best_log_loss': best.value, 'best_trial': best.number,
               'artifact': best.user_attrs['artifact'], 'trials': len(study.trials)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--config', type=Path, default=Path('configs/baselines.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mlflow', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or Path('experiments/optuna') / stamp
    print(json.dumps(tune(json.loads(args.config.read_text()), output, args.trials, args.mlflow), indent=2))


if __name__ == '__main__':
    main()
