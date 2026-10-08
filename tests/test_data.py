import unittest
import tempfile
from pathlib import Path
import pandas as pd
from autoresearch_purchase_propensity.features import add_history
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
        self.assertEqual(out.history_sessions.tolist(), [0, 1, 1, 3, 0])
        self.assertEqual(out.history_purchase_sessions.tolist(), [0, 1, 1, 2, 0])
        self.assertEqual(out.loc[3, 'history_mean_time_on_site'], 150)
        self.assertTrue(pd.isna(out.loc[1, 'history_mean_time_on_site']))
        self.assertEqual(out.loc[3, 'days_since_last_purchase'], 31)
        self.assertTrue(pd.isna(out.loc[4, 'days_since_last_session']))

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
