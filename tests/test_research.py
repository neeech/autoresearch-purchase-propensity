"""Exercise actual Git commits and subprocesses without an API or ML dependencies."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from autoresearch_purchase_propensity.research import EDITABLE, ROOT, RequestLimiter, apply_proposal, git, main, propose, retry_delay, run, train


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

    def entries(self):
        return [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines()]

    def test_resume_keeps_best_history_and_adds_attempts_with_new_provider(self):
        first = run(self.root, self.output, lambda context: self.proposal(.4), experiments=1,
                    provider='gemini', model='first-model')
        baseline = (self.output / 'run_000/training.log').stat().st_mtime_ns
        contexts = []
        proposals = iter([self.proposal(.6), self.proposal(.3)])
        def proposer(context):
            contexts.append({**context, 'history': list(context['history'])})
            return next(proposals)
        summary = run(self.root, self.output, proposer, experiments=2, resume=True,
                      provider='groq', model='second-model')
        entries = self.entries()
        self.assertEqual([e['experiment'] for e in entries], [0, 1, 2, 3])
        self.assertEqual([e['status'] for e in entries], ['baseline', 'keep', 'discard', 'keep'])
        self.assertEqual(contexts[0]['best_score'], .4)
        self.assertEqual(len(contexts[0]['history']), 2)
        self.assertIn('return 0.4', contexts[0]['files'][EDITABLE[1]])
        self.assertEqual(entries[2]['parent_commit'], first['best_commit'])
        self.assertEqual(entries[3]['parent_commit'], first['best_commit'])
        self.assertEqual(entries[1]['provider'], 'gemini')
        self.assertEqual(entries[3]['provider'], 'groq')
        self.assertEqual(summary['best_log_loss'], .3)
        self.assertEqual(baseline, (self.output / 'run_000/training.log').stat().st_mtime_ns)
        self.assertEqual(len(list((self.output / 'executions').glob('*.json'))), 2)
        self.assertEqual(git(self.root, 'status', '--porcelain'), '')

    def test_resume_recovers_history_when_summary_write_was_interrupted(self):
        from autoresearch_purchase_propensity.research import write_json
        def write(path, value):
            if path.name == 'summary.json' and value['best_log_loss'] == .4:
                raise OSError('Simulated checkpoint interruption')
            write_json(path, value)
        with patch('autoresearch_purchase_propensity.research.write_json', side_effect=write):
            with self.assertRaisesRegex(OSError, 'checkpoint interruption'):
                run(self.root, self.output, lambda context: self.proposal(.4), experiments=1)
        summary = run(self.root, self.output, lambda context: self.proposal(.3),
                      experiments=1, resume=True)
        self.assertEqual([e['status'] for e in self.entries()], ['baseline', 'keep', 'keep'])
        self.assertEqual(summary['best_log_loss'], .3)

    def test_resume_records_pending_attempt_and_preserves_its_artifacts(self):
        from autoresearch_purchase_propensity.research import write_json
        summary = run(self.root, self.output, lambda context: None, experiments=0)
        worktree = Path(summary['worktree'])
        apply_proposal(worktree, self.proposal(.2))
        git(worktree, 'add', *EDITABLE)
        git(worktree, 'commit', '-m', 'Unconfirmed candidate')
        write_json(self.output / 'pending.json', {'experiment': 1, 'hypothesis': 'Unconfirmed',
                   'parent_commit': summary['best_commit'], 'commit': git(worktree, 'rev-parse', 'HEAD'),
                   'metrics': None, 'status': 'running'})
        (self.output / 'run_001').mkdir()
        artifact = self.output / 'run_001/training.log'
        artifact.write_text('Partial training')
        result = run(self.root, self.output, lambda context: self.proposal(.4), experiments=1, resume=True)
        self.assertEqual([e['status'] for e in self.entries()], ['baseline', 'interrupted', 'keep'])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(artifact.read_text(), 'Partial training')
        self.assertEqual(self.entries()[2]['parent_commit'], self.initial)

    def test_resume_rejects_changed_frozen_data_and_protocol(self):
        run(self.root, self.output, lambda context: None, experiments=0)
        path = self.output / 'data/train_X.parquet'
        previous = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'frozen data changed'):
            run(self.root, self.output, lambda context: None, resume=True)
        path.write_bytes(previous)
        (self.root / 'program.md').write_text('Different instructions')
        git(self.root, 'add', 'program.md')
        git(self.root, 'commit', '-m', 'Change protocol')
        with self.assertRaisesRegex(ValueError, 'Protocol differs'):
            run(self.root, self.output, lambda context: None, resume=True)
        self.assertEqual(len(self.entries()), 1)

    def test_research_cannot_run_twice_at_the_same_time(self):
        from autoresearch_purchase_propensity.research import research_lock
        with research_lock(self.output):
            with self.assertRaisesRegex(ValueError, 'already running'):
                run(self.root, self.output, lambda context: None, experiments=0)
        self.assertFalse(self.output.exists())
        run(self.root, self.output, lambda context: None, experiments=0)

    def test_interrupt_stops_and_can_be_resumed(self):
        def interrupt(context):
            raise KeyboardInterrupt()
        result = run(self.root, self.output, interrupt, experiments=3)
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(self.entries()[1]['status'], 'interrupted')
        summary = run(self.root, self.output, lambda context: self.proposal(.4), experiments=1, resume=True)
        self.assertEqual(summary['attempts'], 2)
        self.assertEqual(summary['best_log_loss'], .4)

    def test_daily_quota_stops_without_spending_remaining_attempts(self):
        from io import BytesIO
        from urllib.error import HTTPError
        error = HTTPError('https://api.groq.com', 429, 'Too Many Requests', {},
                          BytesIO(b'{"error":{"message":"Rate limit reached for requests per day (RPD)"}}'))
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
                patch('autoresearch_purchase_propensity.research.urlopen', side_effect=error) as request, \
                patch('autoresearch_purchase_propensity.research.time.sleep') as sleep:
            summary = run(self.root, self.output, lambda context: propose(context), experiments=5)
        self.assertEqual(summary['attempts'], 1)
        self.assertEqual(self.entries()[1]['status'], 'quota_exhausted')
        request.assert_called_once()
        # subprocess.wait also uses time.sleep for polling; no API backoff occurred.
        self.assertTrue(all(call.args[0] < 2 for call in sleep.call_args_list))
        result = run(self.root, self.output, lambda context: self.proposal(.4), experiments=1, resume=True)
        self.assertEqual(result['best_log_loss'], .4)

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

    def test_api_failure_is_recorded_and_loop_continues_from_best(self):
        from urllib.error import HTTPError
        calls = []
        def unavailable(context):
            calls.append(context)
            if len(calls) == 1:
                raise HTTPError('https://api.groq.com', 429, 'Quota exceeded', {}, None)
            return self.proposal(.4)

        summary = run(self.root, self.output, unavailable, experiments=2, timeout=30)
        entries = [json.loads(line) for line in (self.output / 'history.jsonl').read_text().splitlines()]
        self.assertEqual([entry['status'] for entry in entries], ['baseline', 'failed', 'keep'])
        self.assertEqual(entries[2]['parent_commit'], self.initial)
        self.assertEqual(summary['best_log_loss'], .4)
        self.assertIn('429', entries[1]['error'])

    def test_api_retry_after_and_backoff_then_success(self):
        from io import BytesIO
        from urllib.error import HTTPError, URLError
        error = HTTPError('https://api.groq.com', 429, 'Rate limit', {'Retry-After': '7'},
                          BytesIO(b'{"error":{"message":"Wait"}}'))
        response = {'choices': [{'message': {'content': json.dumps(self.proposal(.4))}}]}
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
                patch('autoresearch_purchase_propensity.research.urlopen',
                      side_effect=[error, URLError('Connection lost'), BytesIO(json.dumps(response).encode())]) as request, \
                patch('autoresearch_purchase_propensity.research.time.sleep') as sleep:
            result = propose({'instructions': 'Start with baselines'}, 'test-model')
        self.assertEqual(result, self.proposal(.4))
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [7, 4])

    def test_api_retry_budget_is_bounded(self):
        from io import BytesIO
        from urllib.error import HTTPError
        errors = [HTTPError('https://api.groq.com', 503, 'Unavailable', {},
                            BytesIO(b'{"error":{"message":"Temporary failure"}}')) for _ in range(3)]
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
                patch('autoresearch_purchase_propensity.research.urlopen', side_effect=errors) as request, \
                patch('autoresearch_purchase_propensity.research.time.sleep') as sleep:
            with self.assertRaises(HTTPError):
                propose({'instructions': 'Start with baselines'}, 'test-model')
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_retry_after_http_date_and_invalid_header(self):
        with patch('autoresearch_purchase_propensity.research.time.time', return_value=0):
            self.assertEqual(retry_delay('Thu, 01 Jan 1970 00:00:09 GMT', 0), 9)
        self.assertEqual(retry_delay('invalid', 1), 4)
        self.assertEqual(retry_delay('nan', 0), 2)

    def test_generated_json_failure_can_be_retried(self):
        from io import BytesIO
        from urllib.error import HTTPError
        error = HTTPError('https://api.groq.com', 400, 'Bad Request', {},
                          BytesIO(b'{"error":{"code":"json_validate_failed","message":"Invalid generated JSON"}}'))
        response = {'choices': [{'message': {'content': json.dumps(self.proposal(.4))}}]}
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
                patch('autoresearch_purchase_propensity.research.urlopen',
                      side_effect=[error, BytesIO(json.dumps(response).encode())]) as request, \
                patch('autoresearch_purchase_propensity.research.time.sleep') as sleep:
            self.assertEqual(propose({'instructions': 'Start with baselines'}, 'test-model'), self.proposal(.4))
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(2)

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

    def test_llm_http_error_shows_provider_detail_without_api_key(self):
        from io import BytesIO
        from urllib.error import HTTPError
        response = BytesIO(json.dumps({'error': {'message': 'Invalid request for test-key'}}).encode())
        error = HTTPError('https://api.groq.com', 400, 'Bad Request', {}, response)
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
                patch('autoresearch_purchase_propensity.research.urlopen', side_effect=error) as request, \
                patch('autoresearch_purchase_propensity.research.time.sleep') as sleep:
            with self.assertRaises(HTTPError) as raised:
                propose({'instructions': 'Start with baselines'}, 'test-model')
        self.assertIn('400', str(raised.exception))
        self.assertIn('Invalid request', str(raised.exception))
        self.assertNotIn('test-key', str(raised.exception))
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_timeout_kills_actual_subprocess(self):
        path = self.root / 'src/autoresearch_purchase_propensity/train.py'
        path.write_text('import time\ntime.sleep(30)\n')
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            train(self.root, {'data_dir': str(self.root / 'data/processed')}, self.output, timeout=1)
        self.assertLess(time.monotonic() - started, 10)


class RateLimitTests(unittest.TestCase):
    def test_spaces_requests_and_counts_elapsed_training_time(self):
        now = [0.0]
        def advance(seconds):
            now[0] += seconds
        limiter = RequestLimiter(10)
        with patch('autoresearch_purchase_propensity.research.time.monotonic', side_effect=lambda: now[0]), \
                patch('autoresearch_purchase_propensity.research.time.sleep', side_effect=advance) as sleep:
            limiter.wait()
            sleep.assert_not_called()
            now[0] += 2  # Response/training took two seconds.
            limiter.wait()
            sleep.assert_called_once_with(4)
            self.assertEqual(now[0], 6)
            now[0] += 20  # No additional waiting after a slow experiment.
            limiter.wait()
            self.assertEqual(sleep.call_count, 1)
            limiter.wait()
            self.assertEqual(now[0], 32)

    def test_retries_and_next_proposal_share_limit_and_retry_after(self):
        from io import BytesIO
        from urllib.error import HTTPError
        for retry_after, expected in [('2', [0, 6, 12]), ('9', [0, 9, 15])]:
            now, starts = [0.0], []
            proposal = {'hypothesis': 'Test', 'files': {EDITABLE[0]: 'source'}}
            def request(*args, **kwargs):
                starts.append(now[0])
                if len(starts) == 1:
                    raise HTTPError('https://api.groq.com', 429, 'Rate limit', {'Retry-After': retry_after},
                                    BytesIO(b'{"error":{"message":"Wait"}}'))
                return BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(proposal)}}]}).encode())
            def advance(seconds):
                now[0] += seconds
            limiter = RequestLimiter(10)
            with self.subTest(retry_after=retry_after), \
                    patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True), \
                    patch('autoresearch_purchase_propensity.research.time.monotonic', side_effect=lambda: now[0]), \
                    patch('autoresearch_purchase_propensity.research.time.sleep', side_effect=advance), \
                    patch('autoresearch_purchase_propensity.research.urlopen', side_effect=request):
                self.assertEqual(propose({'instructions': 'Test'}, limiter=limiter), proposal)
                self.assertEqual(propose({'instructions': 'Test'}, limiter=limiter), proposal)
            self.assertEqual(starts, expected)

    def test_cli_shares_limiter_and_rejects_invalid_rpm_before_run(self):
        def fake_run(root, output, proposer, experiments, timeout, **kwargs):
            proposer({'instructions': 'First'})
            proposer({'instructions': 'Second'})
            return {'best_log_loss': .5}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}, clear=True), \
                patch('sys.argv', ['autoresearch', '--provider', 'gemini', '--rpm', '10']), \
                patch('autoresearch_purchase_propensity.research.run', side_effect=fake_run), \
                patch('autoresearch_purchase_propensity.research.propose') as proposal, \
                patch('builtins.print'):
            main()
        first, second = proposal.call_args_list
        self.assertIs(first.kwargs['limiter'], second.kwargs['limiter'])
        self.assertEqual(first.kwargs['limiter'].interval, 6)
        for rpm in ['0', '-1', '1.5', 'nan']:
            with self.subTest(rpm=rpm), patch('sys.argv', ['autoresearch', '--rpm', rpm]), \
                    patch('sys.stderr'), patch('autoresearch_purchase_propensity.research.run') as run_loop:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                run_loop.assert_not_called()


class ProviderTests(unittest.TestCase):
    def test_gemini_request_and_multipart_response(self):
        from io import BytesIO
        proposal = {'hypothesis': 'New feature', 'files': {EDITABLE[0]: 'def build_features(frame):\n    return frame\n'}}
        content = json.dumps(proposal)
        response = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'thought': True, 'text': 'Internal reasoning'},
            {'text': content[:20]}, {'text': content[20:]}]}}]}
        context = {'instructions': 'Use only past sessions', 'files': {}, 'history': []}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'gemini-secret'}, clear=True), \
                patch('autoresearch_purchase_propensity.research.urlopen',
                      return_value=BytesIO(json.dumps(response).encode())) as request:
            self.assertEqual(propose(context, provider='gemini'), proposal)
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent')
        self.assertEqual(sent.get_header('X-goog-api-key'), 'gemini-secret')
        payload = json.loads(sent.data)
        self.assertEqual(payload['systemInstruction']['parts'][0]['text'], context['instructions'])
        self.assertEqual(json.loads(payload['contents'][0]['parts'][0]['text']), {'files': {}, 'history': []})
        # TextResponseFormat.MimeType is a protobuf enum, not a MIME string.
        self.assertEqual(payload['generationConfig']['responseFormat']['text']['mimeType'], 'APPLICATION_JSON')
        self.assertNotIn('gemini-secret', sent.full_url + json.dumps(payload))

    def test_gemini_rate_limit_retries_and_redacts_error(self):
        from io import BytesIO
        from urllib.error import HTTPError
        errors = [HTTPError('https://generativelanguage.googleapis.com', 429, 'Too Many Requests',
                            {'Retry-After': '9'}, BytesIO(b'{"error":{"code":429,"message":"Quota for gemini-secret","status":"RESOURCE_EXHAUSTED"}}'))
                  for _ in range(3)]
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'gemini-secret'}, clear=True), \
                patch('autoresearch_purchase_propensity.research.urlopen', side_effect=errors) as request, \
                patch('autoresearch_purchase_propensity.research.time.sleep') as sleep:
            with self.assertRaises(HTTPError) as raised:
                propose({'instructions': 'Test'}, provider='gemini')
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [9, 9])
        self.assertIn('Quota', str(raised.exception))
        self.assertNotIn('gemini-secret', str(raised.exception))

    def test_gemini_blocked_or_truncated_response_reports_reason(self):
        from io import BytesIO
        for response, reason in [({'promptFeedback': {'blockReason': 'SAFETY'}}, 'SAFETY'),
                                 ({'candidates': [{'finishReason': 'MAX_TOKENS'}]}, 'MAX_TOKENS')]:
            with self.subTest(reason=reason), \
                    patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}, clear=True), \
                    patch('autoresearch_purchase_propensity.research.urlopen',
                          return_value=BytesIO(json.dumps(response).encode())):
                with self.assertRaisesRegex(RuntimeError, reason):
                    propose({'instructions': 'Test'}, provider='gemini')

    def test_provider_requires_its_own_key(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'):
                propose({'instructions': 'Test'}, provider='gemini')
        with self.assertRaisesRegex(ValueError, 'Unknown API provider'):
            propose({}, provider='unknown')

    def test_cli_selects_provider_and_model_and_allows_baseline_without_key(self):
        for arguments, env, expected in [
            (['--provider', 'gemini'], {'GEMINI_API_KEY': 'test-key'}, (None, 'gemini')),
            (['--provider', 'gemini', '--model', 'custom-model'], {'GEMINI_API_KEY': 'test-key'}, ('custom-model', 'gemini')),
            ([], {'GROQ_API_KEY': 'test-key'}, (None, 'groq')),
            (['--provider', 'gemini', '--experiments', '0'], {}, None),
        ]:
            def fake_run(root, output, proposer, experiments, timeout, **kwargs):
                if experiments:
                    proposer({'instructions': 'Test'})
                return {'best_log_loss': .5}
            with self.subTest(arguments=arguments), patch.dict(os.environ, env, clear=True), \
                    patch('sys.argv', ['autoresearch', *arguments]), \
                    patch('autoresearch_purchase_propensity.research.run', side_effect=fake_run), \
                    patch('autoresearch_purchase_propensity.research.propose') as proposal, \
                    patch('builtins.print'):
                main()
                if expected:
                    proposal.assert_called_once_with({'instructions': 'Test'}, *expected)
                else:
                    proposal.assert_not_called()

    def test_training_removes_both_api_keys(self):
        from unittest.mock import Mock
        process = Mock()
        process.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'GROQ_API_KEY': 'groq-secret', 'GEMINI_API_KEY': 'gemini-secret'}), \
                patch('autoresearch_purchase_propensity.research.subprocess.Popen', return_value=process) as popen:
            with self.assertRaisesRegex(RuntimeError, 'Training failed'):
                train(Path(directory), {}, Path(directory) / 'run', 10)
        self.assertNotIn('GROQ_API_KEY', popen.call_args.kwargs['env'])
        self.assertNotIn('GEMINI_API_KEY', popen.call_args.kwargs['env'])


if __name__ == '__main__':
    unittest.main()
