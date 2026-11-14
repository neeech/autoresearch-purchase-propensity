import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from autoresearch_purchase_propensity.features import build_native_features
from autoresearch_purchase_propensity.modeling import build_model
from autoresearch_purchase_propensity.marketing import compare


class TreeBaselineTests(unittest.TestCase):
    def test_native_missing_categories_and_no_future_fit(self):
        x = pd.DataFrame({'hits': list(range(1, 21)), 'bounces': [1, 0]*10,
                          'timeOnSite': [np.nan, 20.]*10, 'source': ['a', 'b']*10,
                          'history': [[]]*20})
        prepared = build_native_features(x)
        self.assertEqual(prepared.timeOnSite.isna().sum(), 10)
        self.assertNotIn('history', prepared)
        self.assertNotIn('timeOnSite_missing', prepared)
        model = build_model(x.columns, {'model': 'lightgbm', 'seed': 42,
                            'lightgbm': {'n_estimators': 5, 'min_child_samples': 1}})
        model.fit(x, [0, 1]*10)
        categories = model.named_steps['model'].booster_.pandas_categorical.copy()
        future = x.iloc[:2].copy()
        future['source'] = ['unseen', None]
        self.assertTrue(np.isfinite(model.predict_proba(future)).all())
        self.assertEqual(categories, model.named_steps['model'].booster_.pandas_categorical)
        self.assertFalse(x.source.dtype == 'category')

    def test_direct_spending_is_separate_from_penalties(self):
        frame = pd.DataFrame({'session_row_id': [1, 2], 'target': [1, 0], 'probability': [.8, .2]})
        report, _ = compare(frame, {'main': dict(tp=1, tn=0, fp=5, fn=10)}, contact_cost=2.)
        row = report.iloc[0]
        self.assertEqual(row.direct_spend, 2.)
        self.assertEqual(row.total_penalty, 1.)

    @unittest.skipUnless(importlib.util.find_spec('optuna'), 'Install tuning extra')
    def test_tuner_freezes_only_train_validation_and_preserves_best_config(self):
        from autoresearch_purchase_propensity.tune import tune
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'source'; data.mkdir()
            for split in ('train', 'validation', 'test'):
                for suffix in ('X', 'y', 'metadata'):
                    (data / f'{split}_{suffix}.parquet').write_bytes(b'fixture')
            (data / 'manifest.json').write_text('{}')
            def fake_run(config):
                self.assertEqual(config['model'], 'lightgbm')
                self.assertNotIn('early_stopping_rounds', config['lightgbm'])
                artifact = Path(config['output_root']) / str(config['lightgbm']['num_leaves'])
                artifact.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([{'model': 'candidate', 'log_loss': .3, 'average_precision': .6}]).to_csv(artifact/'results.csv', index=False)
                return artifact
            with patch('autoresearch_purchase_propensity.tune.run', side_effect=fake_run):
                result = tune({'data_dir': str(data), 'seed': 42}, root/'study', trials=2)
            self.assertEqual(result['trials'], 2)
            self.assertFalse(list((root/'study/data').glob('test*')))
            best = json.loads((root/'study/best_config.json').read_text())
            self.assertEqual(best['model'], 'lightgbm')
