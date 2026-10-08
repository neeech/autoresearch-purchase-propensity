import unittest
import numpy as np
import pandas as pd
from autoresearch_purchase_propensity.evaluate import metrics
from autoresearch_purchase_propensity.modeling import build_model


class EvaluationTests(unittest.TestCase):
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
