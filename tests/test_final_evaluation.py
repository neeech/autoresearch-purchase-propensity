import hashlib
from pathlib import Path
import tempfile
import unittest

from autoresearch_purchase_propensity.final_evaluation import verify_files
from autoresearch_purchase_propensity.marketing import compare
import pandas as pd


class FinalEvaluationTests(unittest.TestCase):
    def test_frozen_file_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input').write_bytes(b'original')
            expected = {'input': hashlib.sha256(b'original').hexdigest()}
            verify_files(root, expected)
            (root / 'input').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_files(root, expected)

    def test_test_labels_cannot_reselect_frozen_threshold(self):
        costs = {'main': dict(tp=1, tn=0, fp=5, fn=10)}
        frozen = {'main': .7}
        frame = pd.DataFrame({'session_row_id': [1, 2, 3], 'target': [1, 1, 0],
                              'probability': [.1, .4, .8]})
        report, thresholds = compare(frame, costs, frozen)
        row = report.query("strategy == 'model'").iloc[0]
        self.assertEqual(thresholds, frozen)
        self.assertEqual(row.fp, 1)
        self.assertEqual(row.fn, 2)
        frame.target = [0, 0, 1]
        changed, thresholds = compare(frame, costs, frozen)
        self.assertEqual(thresholds, frozen)
        self.assertEqual(changed.iloc[0].contacts, 1)
