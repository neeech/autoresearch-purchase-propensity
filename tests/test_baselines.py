import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from autoresearch_purchase_propensity.evaluate import metrics
from autoresearch_purchase_propensity.modeling import build_model
from autoresearch_purchase_propensity.features import build_features
from autoresearch_purchase_propensity.train import load_data


class EvaluationTests(unittest.TestCase):
    def test_history_parquet_roundtrip_returns_lists_for_train_and_validation(self):
        first = {'purchased': 1, 'days_ago': 2}
        second = {'purchased': 0, 'days_ago': 5}
        expected = [[], [first], [first, second], []]
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for split in ('train', 'validation'):
                pd.DataFrame({'month': [1] * 4, 'hits': [1, 2, 3, 4],
                              'history': [[], [first], [first, second], None]}).to_parquet(
                    data / f'{split}_X.parquet', index=False)
                pd.DataFrame({'session_row_id': [9, 3, 7, 1], 'target': [0, 1, 0, 1]}).to_parquet(
                    data / f'{split}_y.parquet', index=False)
                pd.DataFrame({'session_row_id': [9, 3, 7, 1]}).to_parquet(
                    data / f'{split}_metadata.parquet', index=False)
                x, y = load_data(data, split)
                self.assertTrue(all(isinstance(events, list) for events in x.history))
                self.assertEqual(x.history.tolist(), expected)
                self.assertEqual(x.hits.tolist(), [1, 2, 3, 4])
                np.testing.assert_array_equal(y, [0, 1, 0, 1])
                # Reproduce the generated feature's `if not events` on empty AND multi-event histories.
                counts = x.history.map(lambda events: 0 if not events else len(events))
                self.assertEqual(counts.tolist(), [0, 1, 2, 0])

    def test_engineered_column_is_used_after_feature_creation(self):
        frame = pd.DataFrame({'hits': [1, 2, 3, 4, 5, 6]})

        def engineered(frame):
            result = build_features(frame)
            result['hits_squared'] = result.hits ** 2
            return result

        config = {'seed': 42, 'min_category_frequency': 2,
                  'logistic': {'solver': 'liblinear', 'C': 1., 'max_iter': 100}}
        with patch('autoresearch_purchase_propensity.modeling.build_features', engineered):
            model = build_model(frame.columns.tolist(), config)
        model.fit(frame, [0, 0, 0, 1, 1, 1])
        names = model.named_steps['features'].get_feature_names_out()
        self.assertIn('numeric__hits_squared', names)
        self.assertEqual(model.predict_proba(frame).shape, (6, 2))

    def test_constant_reference_and_invalid_predictions(self):
        result = metrics([0, 0, 0, 1], [.25]*4)
        self.assertAlmostEqual(result['average_precision'], .25)
        self.assertAlmostEqual(result['log_loss'], -(.75*np.log(.75)+.25*np.log(.25)))
        for bad in [[float('nan')]*4, [1.1]*4, [0.5]*3]:
            with self.assertRaises(ValueError):
                metrics([0, 0, 0, 1], bad)

    def test_unseen_categories_and_no_validation_fit(self):
        frame=pd.DataFrame({'hits':[1,2,3,4,5,6],
                            'source':pd.Series(['a','a','a','b','b',pd.NA],dtype='string'),
                            'timeOnSite':[np.nan,10,20,30,40,50]})
        config={'seed':42,'min_category_frequency':2,
                'logistic':{'solver':'liblinear','C':1.,'max_iter':100}}
        model=build_model(frame.columns.tolist(),config)
        model.fit(frame,[0,0,0,0,1,1])
        before=model.named_steps['features'].named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
        validation=pd.DataFrame({'hits':[999],'source':pd.Series(['unseen'],dtype='string'),'timeOnSite':[np.nan]})
        probability=model.predict_proba(validation)[:,1]
        self.assertTrue(np.isfinite(probability).all())
        after=model.named_steps['features'].named_transformers_['numeric'].named_steps['impute'].statistics_
        np.testing.assert_array_equal(before,after)


if __name__=='__main__':
    unittest.main()
