"""Exercise actual Git commits and subprocesses without an API or ML dependencies."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from autoresearch_purchase_propensity.research import EDITABLE, ROOT, apply_proposal, git, propose, run, train


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        self.output = Path(self.temp.name) / 'trial'
        git(self.root, 'init')
        git(self.root, 'config', 'user.name', 'Research Test')
        git(self.root, 'config', 'user.email', 'research@example.invalid')
        git(self.root, 'config', 'core.autocrlf', 'false')
        package = self.root / 'src/autoresearch_purchase_propensity'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        (package / 'features.py').write_text('def build_features(frame):\n    return frame\n')
        (package / 'modeling.py').write_text('def build_model(columns, config):\n    return 0.5\n')
        (package / 'history.py').write_text('# fixed history\n')
        # A deliberately small training entrypoint lets us test the real subprocess/Git path.
        (package / 'train.py').write_text('''import argparse, csv, json
from pathlib import Path
from .modeling import build_model
parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
data = Path(config['data_dir'])
assert not (data / 'test_X.parquet').exists()
out = Path(config['output_root']) / 'result'
out.mkdir(parents=True)
with (out / 'results.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model', 'log_loss', 'average_precision'])
    writer.writerow(['constant', 0.8, 0.1])
    writer.writerow(['candidate', build_model([], config), 0.5])
(out.parent / 'latest.json').write_text(json.dumps({'run_dir': str(out)}))
''')
        (self.root / 'configs').mkdir()
        (self.root / 'configs/baselines.json').write_text(json.dumps({'data_dir': 'data/processed'}))
        (self.root / 'program.md').write_text((ROOT / 'program.md').read_text(encoding='utf-8'), encoding='utf-8')
        (self.root / '.gitignore').write_text('data/processed/\n__pycache__/\n')
        data = self.root / 'data/processed'
        data.mkdir(parents=True)
        for split in ('train', 'validation', 'test'):
            for suffix in ('X', 'y', 'metadata'):
                (data / f'{split}_{suffix}.parquet').write_bytes(b'frozen input')
        (data / 'manifest.json').write_text('{}')
        git(self.root, 'add', '.')
        git(self.root, 'commit', '-m', 'Initial baselines')
        self.initial = git(self.root, 'rev-parse', 'HEAD')

    def proposal(self, value):
        return {'hypothesis': f'Try {value}',
                'files': {EDITABLE[1]: f'def build_model(columns, config):\n    return {value}\n'}}

    def test_real_commits_keep_discard_tie_failure_and_next_parent(self):
        proposals = [self.proposal(.4), self.proposal(.6), self.proposal(.4),
                     self.proposal("float('nan')"), self.proposal(.3)]
        proposals[2]['files'][EDITABLE[1]] += '# A different candidate with the same score\n'
        contexts = []

        def proposer(context):
            contexts.append(context)
            return proposals.pop(0)

        summary = run(self.root, self.output, proposer, experiments=5, timeout=30)
        history = [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines()]
        self.assertEqual([e['status'] for e in history], ['baseline', 'keep', 'discard', 'discard', 'failed', 'keep'])
        self.assertEqual(history[0]['metrics']['constant']['log_loss'], .8)
        self.assertEqual(history[2]['parent_commit'], history[1]['commit'])
        self.assertEqual(history[5]['parent_commit'], history[1]['commit'])
        self.assertEqual(summary['best_log_loss'], .3)
        self.assertEqual(git(Path(summary['worktree']), 'rev-parse', 'HEAD'), summary['best_commit'])
        self.assertEqual(git(self.root, 'rev-parse', 'HEAD'), self.initial)
        self.assertEqual(git(self.root, 'status', '--porcelain'), '')
        self.assertEqual(len(git(Path(summary['worktree']), 'log', '--oneline').splitlines()), 3)
        self.assertIn('return 0.4', contexts[2]['files'][EDITABLE[1]])
        self.assertIn('baselines', contexts[0]['instructions'])
        self.assertNotIn('test', contexts[0]['data']['splits'])
        self.assertEqual(git(self.root, 'show', history[2]['commit'] + ':' + EDITABLE[1]).splitlines()[-1], '    return 0.6')

    def test_reject_paths_syntax_missing_interface_and_empty_changes(self):
        for proposal in [
            {'hypothesis': 'escape', 'files': {'../escape.py': 'x=1'}},
            {'hypothesis': 'fixed', 'files': {'src/autoresearch_purchase_propensity/history.py': 'x=1'}},
            {'hypothesis': 'syntax', 'files': {EDITABLE[0]: 'def build_features('}},
            {'hypothesis': 'interface', 'files': {EDITABLE[0]: 'x=1'}},
            {'hypothesis': 'empty', 'files': {}},
        ]:
            before = (self.root / EDITABLE[0]).read_text()
            with self.assertRaises((ValueError, SyntaxError)):
                apply_proposal(self.root, proposal)
            self.assertEqual((self.root / EDITABLE[0]).read_text(), before)

    def test_dirty_checkout_rejected_before_creating_worktree(self):
        (self.root / EDITABLE[0]).write_text('# user change')
        with self.assertRaisesRegex(ValueError, 'Commit or stash'):
            run(self.root, self.output, lambda context: self.proposal(.4))
        self.assertFalse(self.output.exists())

    def test_missing_data_rejected_before_creating_branch(self):
        (self.root / 'data/processed/validation_X.parquet').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Run preprocessing first'):
            run(self.root, self.output, lambda context: self.proposal(.4))
        self.assertFalse(self.output.exists())
        self.assertNotIn('autoresearch/', git(self.root, 'branch', '--list'))

    def test_invalid_proposal_keeps_hypothesis_without_a_commit(self):
        proposal = {'hypothesis': 'Change protected history',
                    'files': {'src/autoresearch_purchase_propensity/history.py': '# change'}}
        summary = run(self.root, self.output, lambda context: proposal, experiments=1, timeout=30)
        entry = json.loads((self.output / 'history.jsonl').read_text().splitlines()[1])
        self.assertEqual(entry['hypothesis'], proposal['hypothesis'])
        self.assertEqual(entry['status'], 'failed')
        self.assertIsNone(entry['commit'])
        self.assertEqual(summary['best_commit'], self.initial)

    def test_protected_training_write_stops_and_restores_history(self):
        def tamper(worktree, config, directory, timeout):
            (worktree / 'src/autoresearch_purchase_propensity/history.py').write_text('# altered')
            return {'constant': {'log_loss': .8}, 'candidate': {'log_loss': .1}}

        with patch('autoresearch_purchase_propensity.research.train', side_effect=tamper):
            summary = run(self.root, self.output, lambda context: self.proposal(.4), experiments=3)
        entries = (self.output / 'history.jsonl').read_text().splitlines()
        self.assertEqual(len(entries), 1)
        self.assertEqual(json.loads(entries[0])['status'], 'failed')
        self.assertIsNone(summary['best_log_loss'])
        self.assertEqual((Path(summary['worktree']) / 'src/autoresearch_purchase_propensity/history.py').read_text(), '# fixed history\n')

    def test_api_error_stops_after_baselines_and_is_recorded(self):
        from urllib.error import HTTPError

        def unavailable(context):
            raise HTTPError('https://api.groq.com', 429, 'Quota exceeded', {}, None)

        summary = run(self.root, self.output, unavailable, experiments=3, timeout=30)
        entries = [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines()]
        self.assertEqual([entry['status'] for entry in entries], ['baseline', 'failed'])
        self.assertEqual(summary['best_commit'], self.initial)
        self.assertIn('429', entries[1]['error'])

    def test_llm_request_uses_prompt_and_returns_only_proposal(self):
        from io import BytesIO
        proposal = self.proposal(.4)
        response = {'choices': [{'message': {'content': json.dumps(proposal)}}]}
        context = {'instructions': 'Start with baselines', 'files': {}, 'history': []}
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
                patch('autoresearch_purchase_propensity.research.urlopen',
                      return_value=BytesIO(json.dumps(response).encode())) as request:
            self.assertEqual(propose(context, 'test-model'), proposal)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload['messages'][0]['content'], 'Start with baselines')
        self.assertNotIn('test-key', json.dumps(payload))
        self.assertEqual(context['instructions'], 'Start with baselines')

    def test_timeout_kills_actual_subprocess(self):
        path = self.root / 'src/autoresearch_purchase_propensity/train.py'
        path.write_text('import time\ntime.sleep(30)\n')
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            train(self.root, {'data_dir': str(self.root / 'data/processed')}, self.output, timeout=1)
        self.assertLess(time.monotonic() - started, 10)


if __name__ == '__main__':
    unittest.main()
