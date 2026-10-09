import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from autoresearch_purchase_propensity.research import ALLOWED, MODEL, ROOT, experiment, is_better, run, validate
from autoresearch_purchase_propensity.research_agent import ResearchAgent


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.code = {name: (ROOT / name).read_text() for name in ALLOWED}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def proposal(self, code=None):
        code = code or self.code
        config = json.loads(code[MODEL])
        config['params']['C'] *= 2
        return {'hypothesis': 'Reduce regularization', 'reasoning': 'Try a larger C',
                'changes': [{'file': MODEL, 'old': code[MODEL], 'new': json.dumps(config)}]}

    def test_protected_paths_and_commands(self):
        self.assertNotEqual(validate(self.code, self.proposal()), self.code)
        for name in ('../escape.json', 'src/autoresearch_purchase_propensity/evaluate.py'):
            proposal = self.proposal()
            proposal['changes'][0]['file'] = name
            with self.assertRaises(ValueError):
                validate(self.code, proposal)
        for text in ('import os', '{"model":"logistic","params":{"n_jobs":99},"min_category_frequency":20}'):
            proposal = self.proposal()
            proposal['changes'][0]['new'] = text
            with self.assertRaises(ValueError):
                validate(self.code, proposal)

    def test_loop_keeps_best_and_records_failures(self):
        observed = []
        parent = self

        class Agent:
            def propose(self, context):
                observed.append(context['code'])
                return parent.proposal(context['code'])

        results = [np.array([.5]), np.array([.4]), np.array([.6]),
                   subprocess.TimeoutExpired('worker', 1), np.array([.4])]
        with patch('autoresearch_purchase_propensity.research.load_data', return_value=(pd.DataFrame({'hits':[1]}), [0])), \
             patch('autoresearch_purchase_propensity.research.experiment', side_effect=results), \
             patch('autoresearch_purchase_propensity.research.metrics', side_effect=lambda y,p: {'log_loss':float(p[0])}):
            best = run(Agent(), self.code, self.output, self.output, self.output, 4)
        history = [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines()]
        self.assertEqual(best, .4)
        self.assertEqual([e['status'] for e in history], ['baseline', 'keep', 'discard', 'timeout', 'discard'])
        self.assertEqual(observed[1], observed[2])
        self.assertEqual(observed[2], observed[3])
        self.assertEqual(json.loads((self.output / 'best/autoresearch_model.json').read_text())['params']['C'], 2)

    def test_timeout_terminates_process_group(self):
        with patch('subprocess.Popen') as spawn, patch('os.killpg') as kill:
            spawn.return_value.pid = 12345
            spawn.return_value.wait.side_effect = [subprocess.TimeoutExpired('worker', 1), 0]
            with self.assertRaises(subprocess.TimeoutExpired):
                experiment(self.code, self.output, self.output / 'run', self.output, 1)
        kill.assert_called_once()

    def test_feature_operations_preserve_rows(self):
        from autoresearch_purchase_propensity.research_features import transform
        frame = pd.DataFrame({'hits': [4, 2], 'pageviews': [2, 0]}, index=[3, 1])
        config = {'add': [{'name': 'ratio', 'op': 'ratio', 'columns': ['hits', 'pageviews']}], 'drop': []}
        result = transform(frame, config)
        self.assertTrue(result.index.equals(frame.index))
        self.assertEqual(result.ratio.iloc[0], 2)
        self.assertTrue(pd.isna(result.ratio.iloc[1]))
        self.assertNotIn('ratio', frame)

    def test_groq_adapter_without_network(self):
        with patch.dict('os.environ', {'GROQ_API_KEY':'test-key'}), \
             patch('autoresearch_purchase_propensity.research_agent.urlopen') as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps({
                'choices': [{'message': {'content': json.dumps(self.proposal())}}]}).encode()
            result = ResearchAgent().propose({'code': self.code})
        self.assertEqual(result, self.proposal())
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body['model'], 'openai/gpt-oss-120b')
        self.assertNotIn('test-key', json.dumps(body))
        self.assertEqual(request.call_args.args[0].get_header('User-agent'), 'purchase-autoresearch/0.1')

    def test_api_error_details_are_safe(self):
        from io import BytesIO
        from urllib.error import HTTPError
        response = BytesIO(b'{"error":{"message":"Model forbidden for test-key"}}')
        error = HTTPError('https://api.groq.com', 403, 'Forbidden', {}, response)
        with patch.dict('os.environ', {'GROQ_API_KEY':'test-key'}), \
             patch('autoresearch_purchase_propensity.research_agent.urlopen', side_effect=error):
            with self.assertRaises(HTTPError) as raised:
                ResearchAgent().check()
        self.assertIn('Model forbidden', str(raised.exception))
        self.assertNotIn('test-key', str(raised.exception))

    def test_selection_directions_and_ties(self):
        self.assertTrue(is_better(.4, .5))
        self.assertTrue(is_better(.6, .5, maximize=True))
        self.assertFalse(is_better(.5, .5))

    def test_api_quota_error_is_recorded_and_stops(self):
        from urllib.error import HTTPError
        from unittest.mock import Mock
        agent = Mock()
        agent.propose.side_effect = HTTPError('https://api.groq.com', 429, 'Quota exceeded', {}, None)
        with patch('autoresearch_purchase_propensity.research.load_data', return_value=(pd.DataFrame({'hits':[1]}), [0])), \
             patch('autoresearch_purchase_propensity.research.experiment', return_value=np.array([.5])), \
             patch('autoresearch_purchase_propensity.research.metrics', return_value={'log_loss':.5}):
            run(agent, self.code, self.output, self.output, self.output, 30)
        self.assertEqual(agent.propose.call_count, 1)
        history = [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines()]
        self.assertIn('429', history[-1]['error'])
        self.assertEqual(history[-1]['status'], 'failed')
        self.assertEqual(len(history), 2)


if __name__ == '__main__':
    unittest.main()
