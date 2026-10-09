"""Fixed local training entrypoint; candidate JSON never becomes executable code."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

from autoresearch_purchase_propensity.modeling import build_model
from autoresearch_purchase_propensity.research_features import transform
from autoresearch_purchase_propensity.train import load_data


def run(data, directory):
    data, directory = Path(data), Path(directory)
    x, y = load_data(data, 'train')
    config = json.loads((directory / 'autoresearch_model.json').read_text())
    features = json.loads((directory / 'autoresearch_features.json').read_text())
    transformed = transform(x.copy(), features)
    model = build_model(transformed.columns.tolist(), {
        'seed': 42, 'min_category_frequency': config['min_category_frequency'],
        'logistic': {'solver': 'liblinear', 'max_iter': 500},
    })
    factories = {'random_forest': RandomForestClassifier,
                 'extra_trees': ExtraTreesClassifier}
    if config['model'] == 'lightgbm':
        from lightgbm import LGBMClassifier
        model.steps[-1] = ('model', LGBMClassifier(
            **config['params'], random_state=42, n_jobs=4, verbosity=-1))
    elif config['model'] in factories:
        model.steps[-1] = ('model', factories[config['model']](
            **config['params'], random_state=42, n_jobs=4))
    else:
        model.steps[-1][1].set_params(**config['params'])
    model.steps.insert(0, ('candidate', FunctionTransformer(
        transform, kw_args={'config': features}, validate=False)))
    model.fit(x, y)
    joblib.dump(model, directory / 'model.joblib')
    validation = pd.read_parquet(data / 'validation_X.parquet').drop(columns='month')
    np.save(directory / 'predictions.npy', model.predict_proba(validation)[:, 1], allow_pickle=False)


if __name__ == '__main__':
    run(*sys.argv[1:])
