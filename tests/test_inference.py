import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference.server import create_app


class InferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / 'data_manifest.json').write_text(json.dumps({'features': ['hits', 'source', 'history', 'month']}))
        self.model = Mock()
        self.model.named_steps = {'features': Mock(transformers_=[('numeric', None, ['hits'])])}
        self.model.predict_proba.side_effect = lambda x: np.column_stack([np.full(len(x), .7), np.full(len(x), .3)])
        old_path = sys.path.copy()
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        with patch('inference.server.joblib.load', return_value=self.model):
            self.app = create_app(root)

    def request(self, body=b'', path='/invocations', method='POST', content_type='application/jsonlines'):
        response = []
        result = b''.join(self.app({'PATH_INFO': path, 'REQUEST_METHOD': method,
            'CONTENT_TYPE': content_type, 'CONTENT_LENGTH': str(len(body)),
            'wsgi.input': io.BytesIO(body)}, lambda status, headers: response.append(status)))
        return response[0], result

    def test_ping_and_ordered_multiline_predictions(self):
        self.assertEqual(self.request(path='/ping', method='GET'), ('200 OK', b''))
        body = b'{"hits":2,"source":"a","history":[]}\n{"hits":null,"source":null,"history":[]}\n'
        status, result = self.request(body)
        self.assertEqual(status, '200 OK')
        self.assertEqual(len(result.splitlines()), 2)
        frame = self.model.predict_proba.call_args.args[0]
        self.assertEqual(str(frame.hits.dtype), 'float64')
        self.assertTrue(np.isnan(frame.hits.iloc[1]))
        self.assertNotIn('month', frame)

    def test_native_pipeline_without_column_transformer(self):
        self.model.named_steps = {'prepare': Mock(), 'model': Mock()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data_manifest.json').write_text(json.dumps({
                'features': ['hits', 'timeOnSite', 'source', 'history']}))
            with patch('inference.server.joblib.load', return_value=self.model):
                self.app = create_app(root)
            body = b'{"hits":2,"timeOnSite":null,"source":null,"history":[]}\n'
            self.assertEqual(self.request(body)[0], '200 OK')
            frame = self.model.predict_proba.call_args.args[0]
            self.assertEqual(str(frame.timeOnSite.dtype), 'float64')
            self.assertTrue(frame.timeOnSite.isna().all())

    def test_rejects_wrong_schema_outcomes_and_content_type(self):
        for body in (b'bad json', b'', b'{"hits":2}',
                     b'{"hits":2,"source":"a","history":[],"target":1}',
                     b'{"hits":2,"source":"a","history":null}'):
            self.assertEqual(self.request(body)[0], '400 Bad Request')
        self.assertEqual(self.request(b'{}', content_type='text/csv')[0], '415 Unsupported Media Type')
        self.model.predict_proba.assert_not_called()

    def test_invalid_model_output_is_server_error(self):
        self.model.predict_proba.side_effect = lambda x: np.array([[0., np.nan]])
        self.assertEqual(self.request(b'{"hits":2,"source":"a","history":[]}')[0], '500 Internal Server Error')
