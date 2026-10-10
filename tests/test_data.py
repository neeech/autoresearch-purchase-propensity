import unittest
import json
import tempfile
import sys
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from autoresearch_purchase_propensity.history import add_history
from autoresearch_purchase_propensity.preprocess import temporal_split, load_sessions


class HistoryTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'fullVisitorID': ['a', 'a', 'a', 'a', 'b'],
                             'date': pd.to_datetime(['2017-04-30', '2017-05-01', '2017-05-01', '2017-06-01', '2017-06-01']),
                             'session_row_id': range(5), 'target': [1, 0, 1, 0, 0],
                             'hits': [10, 20, 30, 40, 1], 'pageviews': [5, 10, 15, 20, 1],
                             'timeOnSite': [None, 100, 200, 300, None]})

    def test_same_day_and_cross_split_history(self):
        out = add_history(self.fixture()).set_index('session_row_id')
        self.assertEqual(out.history.map(len).tolist(), [0, 1, 1, 3, 0])
        self.assertEqual(out.loc[1, 'history'], out.loc[2, 'history'])
        events = out.loc[3, 'history']
        self.assertEqual([event['hits'] for event in events], [10, 20, 30])
        self.assertEqual([event['purchased'] for event in events], [1, 0, 1])
        self.assertEqual([event['days_ago'] for event in events], [32, 31, 31])
        self.assertIsNone(events[0]['timeOnSite'])
        self.assertFalse(any(name.startswith('history_') for name in out.columns))
        for event in events:
            self.assertTrue({'target', 'fullVisitorID', 'date', 'session_row_id',
                             'transactions', 'transactionRevenue'}.isdisjoint(event))

    def test_current_and_same_day_outcomes_are_unavailable(self):
        original = self.fixture()
        changed = original.copy()
        changed.loc[1:2, ['target', 'hits']] = [0, 999]
        before, after = add_history(original), add_history(changed)
        self.assertEqual(before.history.iloc[1:3].tolist(), after.history.iloc[1:3].tolist())

    def test_unsorted_input_preserves_rows_and_has_correct_history(self):
        frame = self.fixture().iloc[[3, 0, 4, 2, 1]]
        result = add_history(frame)
        self.assertTrue(result.index.equals(frame.index))
        self.assertEqual(result.session_row_id.tolist(), frame.session_row_id.tolist())
        self.assertEqual(result.history.map(len).tolist(), [3, 0, 0, 1, 1])

    def test_future_changes_do_not_change_past_features(self):
        original = self.fixture()
        changed = original.copy()
        changed.loc[3, ['target', 'hits', 'timeOnSite']] = [1, 999, 999]
        pd.testing.assert_frame_equal(add_history(original).iloc[:3], add_history(changed).iloc[:3])

    def test_split_boundaries_and_identity(self):
        parts = temporal_split(add_history(self.fixture()), '2017-05-01', '2017-06-01')
        self.assertEqual([len(p) for p in parts.values()], [1, 2, 2])
        self.assertEqual(len(set(pd.concat(parts.values()).session_row_id)), 5)


class SourceSemanticsTests(unittest.TestCase):
    def test_preprocessing_exports_events_and_keeps_history_flag_in_metadata(self):
        from autoresearch_purchase_propensity import preprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'source.csv'
            raw.write_text('fixture')
            config = {'raw_path': str(raw), 'output_dir': str(root / 'processed'),
                      'validation_start': '2017-05-01',
                      'test_start': '2017-06-01', 'missing_transactions_mean_zero': True}
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            exported = {}

            def capture(frame, path, index):
                exported[Path(path).name] = frame.copy()

            with patch.object(preprocess, 'load_sessions', return_value=HistoryTests().fixture()), \
                    patch.object(pd.DataFrame, 'to_parquet', capture), \
                    patch.object(sys, 'argv', ['preprocess', '--config', str(config_path)]), \
                    patch.dict(sys.modules, {'autoresearch_purchase_propensity.features': None}):
                preprocess.main()
            x = exported['validation_X.parquet']
            self.assertIn('history', x)
            self.assertNotIn('has_history', x)
            self.assertFalse(any(name.startswith('history_') for name in x.columns))
            self.assertEqual(exported['validation_metadata.parquet'].has_history.tolist(), [1, 1])
            self.assertEqual(exported['test_metadata.parquet'].has_history.tolist(), [1, 0])
            manifest = json.loads((root / 'processed/manifest.json').read_text())
            self.assertIn('days_ago', manifest['history_fields'])
            self.assertIn('purchased', manifest['history_fields'])

    def test_preprocessing_does_not_depend_on_editable_features(self):
        # No experiment can change raw normalization or the temporal history import.
        from autoresearch_purchase_propensity import preprocess
        self.assertEqual(preprocess.add_history.__module__, 'autoresearch_purchase_propensity.history')
        with patch.dict(sys.modules, {'autoresearch_purchase_propensity.features': None}):
            self.test_null_bounce_means_zero_but_duration_remains_unknown()

    def test_null_bounce_means_zero_but_duration_remains_unknown(self):
        header = 'fullVisitorID,visitNumber,date,bounces,hits,pageviews,timeOnSite,transactionRevenue,transactions,source,channelGrouping,browser,deviceCategory,country,city\n'
        rows = '00123,1,20170101,1,1,1,,,,google,Organic Search,Chrome,desktop,Argentina,Cordoba\n00123,2,20170102,,2,2,,,,google,Organic Search,Chrome,desktop,Argentina,Cordoba\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.csv'
            path.write_text(header + rows)
            data = load_sessions(path)
        self.assertEqual(data.bounces.tolist(), [1, 0])
        self.assertNotIn('bounces_missing', data)
        self.assertTrue(data.timeOnSite.isna().all())
        self.assertEqual(data.fullVisitorID.tolist(), ['00123', '00123'])


if __name__ == '__main__':
    unittest.main()
