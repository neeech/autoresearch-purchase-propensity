import itertools
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from autoresearch_purchase_propensity.marketing import compare, threshold_curve


class MarketingTests(unittest.TestCase):
    def test_exact_search_matches_brute_force_with_ties_and_endpoints(self):
        costs = dict(tp=1, tn=0, fp=5, fn=10)
        p = np.array([0, .3, .3, 1.])
        for labels in itertools.product([0, 1], repeat=4):
            y = np.array(labels)
            curve = threshold_curve(y, p, costs)
            for row in curve.itertuples():
                pred = p >= row.threshold
                expected = (sum(pred & (y == 1)) + 5*sum(pred & (y == 0)) + 10*sum(~pred & (y == 1))) / 4
                self.assertEqual(row.mean_cost, expected)
            self.assertEqual(curve.iloc[0].contact_rate, 0)
            self.assertEqual(curve.iloc[-1].contact_rate, 1)

    def test_frozen_threshold_does_not_follow_evaluation_labels(self):
        frame = pd.DataFrame({'session_row_id': [1, 2], 'target': [0, 1], 'probability': [.2, .8]})
        costs = {'main': dict(tp=1, tn=0, fp=5, fn=10)}
        _, thresholds = compare(frame, costs)
        frame.target = [1, 0]
        report, frozen = compare(frame, costs, thresholds)
        self.assertEqual(frozen, thresholds)
        self.assertEqual(report.iloc[0].fp, 1)
        self.assertEqual(report.iloc[0].fn, 1)
        frame.session_row_id = [1, 1]
        with self.assertRaises(ValueError):
            compare(frame, costs)

    def test_training_exports_aligned_probabilities_for_both_baselines(self):
        from autoresearch_purchase_propensity.train import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{}')
            for split in ('train', 'validation'):
                pd.DataFrame({'month': [5]*4, 'hits': [1, 2, 3, 4]}).to_parquet(root / f'{split}_X.parquet')
                pd.DataFrame({'session_row_id': [8, 2, 9, 1], 'target': [0, 0, 1, 1]}).to_parquet(root / f'{split}_y.parquet')
                pd.DataFrame({'session_row_id': [8, 2, 9, 1]}).to_parquet(root / f'{split}_metadata.parquet')
            config = json.loads(Path('configs/baselines.json').read_text())
            config.update(data_dir=str(root), output_root=str(root / 'runs'))
            output = run(config)
            for name in ('candidate', 'constant'):
                frame = pd.read_csv(output / f'{name}_validation_predictions.csv')
                self.assertEqual(frame.session_row_id.tolist(), [8, 2, 9, 1])
                self.assertEqual(frame.target.tolist(), [0, 0, 1, 1])
                self.assertTrue(frame.probability.between(0, 1).all())
            np.testing.assert_allclose(pd.read_csv(output / 'constant_validation_predictions.csv').probability, .5)
