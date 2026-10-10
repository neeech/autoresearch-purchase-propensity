import unittest

import numpy as np
import pandas as pd

from autoresearch_purchase_propensity.features import build_features


class FeatureTests(unittest.TestCase):
    def test_baseline_preserves_rows_and_does_not_force_history_features(self):
        frame = pd.DataFrame({'hits': [2, 1], 'pageviews': [2, np.nan],
                              'timeOnSite': [np.nan, 10],
                              'history': [[{'purchased': 1, 'days_ago': 2}], []],
                              'source': pd.Series(['google', pd.NA], dtype='string', index=[5, 2])},
                             index=[5, 2])
        original = frame.copy(deep=True)
        result = build_features(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertTrue(result.index.equals(frame.index))
        self.assertNotIn('history', result)
        self.assertFalse(any(name.startswith('history_') for name in result.columns))
        self.assertEqual(result.pageviews_missing.tolist(), [0, 1])
        self.assertEqual(result.timeOnSite_missing.tolist(), [1, 0])
        self.assertTrue(pd.isna(result.source.iloc[1]))

    def test_visible_events_allow_arbitrary_time_windows(self):
        events = [{'purchased': 1, 'days_ago': 2}, {'purchased': 1, 'days_ago': 20}]
        frame = pd.DataFrame({'history': [events, []], 'hits': [2, 1]})
        out = build_features(frame)
        # This is an example experiment; these choices are not defined by history.py.
        out['purchases_7d'] = frame.history.map(
            lambda history: sum(event['purchased'] for event in history if event['days_ago'] <= 7))
        out['purchases_30d'] = frame.history.map(
            lambda history: sum(event['purchased'] for event in history if event['days_ago'] <= 30))
        self.assertEqual(out.purchases_7d.tolist(), [1, 0])
        self.assertEqual(out.purchases_30d.tolist(), [2, 0])


if __name__ == '__main__':
    unittest.main()
