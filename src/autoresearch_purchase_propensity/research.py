"""A small loop: propose two-file edits, commit, train, keep or discard."""
import argparse
import ast
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
EDITABLE = ('src/autoresearch_purchase_propensity/features.py',
            'src/autoresearch_purchase_propensity/modeling.py')
DEFAULT_MODELS = {'groq': 'openai/gpt-oss-120b', 'gemini': 'gemini-3.8-flash'}
API_KEYS = {'groq': 'GROQ_API_KEY', 'gemini': 'GEMINI_API_KEY'}


def git(directory, *args):
    return subprocess.check_output(
        ['git', '-C', str(directory), *args], text=True, encoding='utf-8').strip()


def retry_delay(retry_after, attempt):
    """Honor seconds or an HTTP date; otherwise use a short exponential backoff."""
    fallback = 2 ** (attempt + 1)
    if retry_after is None:
        return fallback
    try:
        delay = float(retry_after)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return max(0, delay) if math.isfinite(delay) else fallback


class RequestLimiter:
    """Space request starts evenly; share one instance across the whole run."""

    def __init__(self, rpm):
        if not isinstance(rpm, int) or rpm < 1:
            raise ValueError('RPM must be a positive integer')
        self.interval = 60 / rpm
        self.last_request = None

    def wait(self):
        if self.last_request is not None:
            remaining = self.last_request + self.interval - time.monotonic()
            if remaining > 0:
                print(f'  RPM limit: waiting {remaining:g}s', flush=True)
            while remaining > 0:
                time.sleep(min(remaining, 60))
                remaining = self.last_request + self.interval - time.monotonic()
        self.last_request = time.monotonic()


class DailyQuotaExceeded(RuntimeError):
    """The provider explicitly reported an exhausted daily quota."""


def propose(context, model=None, provider='groq', limiter=None):
    """Request a proposal, with up to three attempts for transient API failures."""
    if provider not in API_KEYS:
        raise ValueError(f'Unknown API provider: {provider}')
    model = model or DEFAULT_MODELS[provider]
    key_name = API_KEYS[provider]
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f'Set {key_name}, or use --proposals for saved proposals')
    user_content = json.dumps({name: value for name, value in context.items() if name != 'instructions'})
    headers = {'Content-Type': 'application/json', 'User-Agent': 'purchase-autoresearch/0.1'}
    if provider == 'groq':
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers['Authorization'] = f'Bearer {key}'
        payload = {'model': model, 'temperature': 0.2, 'max_completion_tokens': 8192,
                   'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'system', 'content': context['instructions']},
                                {'role': 'user', 'content': user_content}]}
    else:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe="")}:generateContent'
        headers['x-goog-api-key'] = key
        schema = {'type': 'object', 'required': ['hypothesis', 'files'],
                  'additionalProperties': False,
                  'properties': {'hypothesis': {'type': 'string'},
                                 'files': {'type': 'object',
                                           'additionalProperties': False,
                                           'properties': {name: {'type': 'string'} for name in EDITABLE}}}}
        payload = {'systemInstruction': {'parts': [{'text': context['instructions']}]},
                   'contents': [{'role': 'user', 'parts': [{'text': user_content}]}],
                   'generationConfig': {'maxOutputTokens': 16384,
                                        'responseFormat': {'text': {'mimeType': 'APPLICATION_JSON',
                                                                    'schema': schema}}}}
    request = Request(url, json.dumps(payload).encode(), headers)
    for attempt in range(3):
        retry_after = None
        try:
            if limiter is not None:
                limiter.wait()
            with urlopen(request, timeout=180) as response:
                result = json.load(response)
            if provider == 'groq':
                content = result['choices'][0]['message']['content']
            else:
                candidates = result.get('candidates', [])
                if not candidates:
                    reason = result.get('promptFeedback', {}).get('blockReason', 'No candidates returned')
                    raise RuntimeError(f'Gemini returned no proposal: {reason}')
                candidate = candidates[0]
                finish = candidate.get('finishReason')
                if finish != 'STOP':
                    raise RuntimeError(f'Gemini returned an incomplete proposal: {finish or "Unknown finish reason"}')
                content = ''.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                                  if not part.get('thought'))
            return json.loads(content)
        except HTTPError as error:
            body = error.read(4096).decode('utf-8', errors='replace')
            try:
                detail = json.loads(body)['error']
                message = detail['message']
                code = detail.get('code')
            except (ValueError, KeyError, TypeError, AttributeError):
                message, code = body or 'No error detail returned', None
            error.msg = f'{error.reason}: {str(message).replace(key, "[REDACTED]")}'
            daily = any(marker in body.lower() for marker in ('perday', 'per_day', 'per day', 'daily'))
            if error.code == 429 and daily:
                raise DailyQuotaExceeded(str(error)) from error
            retryable = error.code in (408, 409, 429, 498) or 500 <= error.code <= 599
            retryable |= error.code == 400 and (code == 'json_validate_failed'
                                                or 'failed to generate json' in str(message).lower())
            if not retryable or attempt == 2:
                raise
            retry_after = (error.headers or {}).get('Retry-After')
            reason = str(error)
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
            if attempt == 2:
                raise
            reason = str(error).replace(key, '[REDACTED]')
        delay = retry_delay(retry_after, attempt)
        print(f'  API retry {attempt + 1}/2 in {delay:g}s: {reason}', flush=True)
        time.sleep(delay)


def apply_proposal(worktree, proposal):
    """Validate all files before writing any. Reject path escapes and invalid Python."""
    if not isinstance(proposal, dict) or set(proposal) != {'hypothesis', 'files'}:
        raise ValueError('Expected hypothesis and files')
    if not isinstance(proposal['hypothesis'], str) or not proposal['hypothesis'].strip():
        raise ValueError('A hypothesis is required')
    files = proposal['files']
    if not isinstance(files, dict) or not files or set(files) - set(EDITABLE):
        raise ValueError('Only features.py and modeling.py may change')
    for name, source in files.items():
        if not isinstance(source, str) or len(source) > 100_000:
            raise ValueError('Expected Python source, maximum 100 KB per file')
        tree = ast.parse(source)
        interface = 'build_features' if name.endswith('features.py') else 'build_model'
        if not any(isinstance(node, ast.FunctionDef) and node.name == interface for node in tree.body):
            raise ValueError(f'Missing {interface}')
    if all((worktree / name).read_text(encoding='utf-8') == source for name, source in files.items()):
        raise ValueError('Proposal has no changes')
    for name, source in files.items():
        (worktree / name).write_text(source, encoding='utf-8')


def fingerprints(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for path in paths}


def train(worktree, config, directory, timeout):
    """Run the fixed entrypoint in a fresh interpreter; kill its tree on timeout."""
    directory.mkdir()
    config = {**config, 'output_root': str(directory / 'artifacts')}
    config_path = directory / 'config.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(worktree / 'src'), 'PYTHONNOUSERSITE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    for key_name in API_KEYS.values():
        env.pop(key_name, None)
    command = [sys.executable, '-m', 'autoresearch_purchase_propensity.train', '--config', str(config_path)]
    options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    with (directory / 'training.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=worktree, env=env, stdout=log, stderr=log, **options)
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if process.poll() is None:
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            raise
    if returncode:
        raise RuntimeError(f'Training failed; see {directory / "training.log"}')
    latest = json.loads((directory / 'artifacts/latest.json').read_text(encoding='utf-8'))
    with (Path(latest['run_dir']) / 'results.csv').open(encoding='utf-8') as file:
        rows = {row['model']: row for row in csv.DictReader(file)}
    values = {name: {metric: float(rows[name][metric]) for metric in ('log_loss', 'average_precision')}
              for name in ('constant', 'candidate')}
    if not all(math.isfinite(value) for scores in values.values() for value in scores.values()):
        raise ValueError('Non-finite evaluation')
    if any(scores['log_loss'] < 0 or not 0 <= scores['average_precision'] <= 1 for scores in values.values()):
        raise ValueError('Invalid evaluation range')
    return values


def write_json(path, value):
    """Replace a checkpoint only after its complete contents reach disk."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def save_history(output, history):
    path = output / 'history.jsonl'
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for entry in history:
            file.write(json.dumps(entry, allow_nan=False) + '\n')
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


@contextmanager
def research_lock(output):
    """An OS lock is released even when the process is killed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ('.' + output.name + '.lock')).open('a+b') as file:
        file.seek(0)
        if os.name == 'nt':
            import msvcrt
            if file.read(1) == b'':
                file.write(b'0')
                file.flush()
            file.seek(0)
            lock = lambda: msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (file.seek(0), msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            lock = lambda: fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(file, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise ValueError('This research is already running') from error
        try:
            yield
        finally:
            unlock()


def run(root, output, proposer, experiments=10, timeout=180, *, resume=False,
        provider=None, model=None, rpm=None):
    """Continue an isolated research branch without changing the user's checkout."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if git(root, 'status', '--porcelain'):
        raise ValueError('Commit or stash pending changes before starting the loop')
    with research_lock(output):
        return run_locked(root, output, proposer, experiments, timeout, resume, provider, model, rpm)


def run_locked(root, output, proposer, experiments, timeout, resume, provider, model, rpm):
    if resume:
        protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
        worktree = output / 'worktree'
        data = output / 'data'
        branch = protocol['branch']
        config = protocol['config']
        hashes = protocol['fixed_sha256']
        fixed = [Path(name) for name in hashes]
        if not hashes or any(value is None for value in hashes.values()) or fingerprints(fixed) != hashes:
            raise ValueError('Protected files or frozen data changed; cannot resume')
        if git(worktree, 'branch', '--show-current') != branch:
            raise ValueError('Research worktree is on a different branch')
        # The frozen checkout defines the protocol; reject changed local instructions/config/code.
        for path in fixed:
            if path.is_relative_to(worktree):
                relative = path.relative_to(worktree)
                if fingerprints([root / relative])[str(root / relative)] != hashes[str(path)]:
                    raise ValueError(f'Protocol differs from the current checkout: {relative}')
        history_path = output / 'history.jsonl'
        history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()] if history_path.exists() else []
        confirmed = [entry for entry in history if entry['status'] in ('baseline', 'keep')]
        if not confirmed:
            raise ValueError('No confirmed baseline to resume; start a new research run')
        best = confirmed[-1]
        best_commit = best['commit']
        best_score = best['metrics']['candidate']['log_loss']
        best_run = str(output / f"run_{best['experiment']:03d}")
        git(worktree, 'cat-file', '-e', best_commit + '^{commit}')
        changed = set(git(worktree, 'diff', 'HEAD', '--name-only').splitlines())
        untracked = git(worktree, 'ls-files', '--others', '--exclude-standard')
        if changed - set(EDITABLE) or untracked:
            raise ValueError('Unexpected changes in research worktree; inspect before resuming')
        pending = output / 'pending.json'
        if pending.exists():
            entry = json.loads(pending.read_text(encoding='utf-8'))
            if not any(item['experiment'] == entry['experiment'] for item in history):
                entry.update(status='interrupted', error='Execution interrupted before confirming this attempt',
                             timestamp=datetime.now(timezone.utc).isoformat())
                history.append(entry)
                save_history(output, history)
            pending.unlink()
        for path in (output / 'executions').glob('*.json'):
            previous = json.loads(path.read_text(encoding='utf-8'))
            if previous['status'] == 'running':
                previous['status'] = 'interrupted'
                write_json(path, previous)
        git(worktree, 'reset', '--hard', best_commit)
        manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
        first = max(entry['experiment'] for entry in history) + 1
    else:
        config = json.loads((root / 'configs/baselines.json').read_text(encoding='utf-8'))
        source_data = root / config['data_dir']
        inputs = [source_data / f'{split}_{suffix}.parquet'
                  for split in ('train', 'validation') for suffix in ('X', 'y', 'metadata')]
        inputs.append(source_data / 'manifest.json')
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing:
            raise FileNotFoundError('Run preprocessing first; missing inputs: ' + ', '.join(missing))
        manifest = json.loads((source_data / 'manifest.json').read_text(encoding='utf-8'))
        output.mkdir(parents=True, exist_ok=False)
        branch = 'autoresearch/' + output.name
        worktree = output / 'worktree'
        git(root, 'worktree', 'add', '-b', branch, str(worktree), 'HEAD')
        data = output / 'data'
        data.mkdir()
        # Freeze only train and validation. Do not copy test or regenerate features.
        for path in inputs:
            shutil.copyfile(path, data / path.name)
        config['data_dir'] = str(data)
        ignored = ('data/',)
        fixed = [worktree / name for name in git(worktree, 'ls-files').splitlines()
                 if name not in EDITABLE and not name.startswith(ignored)]
        fixed += list(data.iterdir())
        hashes = fingerprints(fixed)
        best_commit = git(worktree, 'rev-parse', 'HEAD')
        protocol = {'branch': branch, 'initial_commit': best_commit, 'editable': EDITABLE,
                    'primary_metric': 'validation log_loss', 'direction': 'minimize',
                    'experiments': experiments, 'timeout': timeout, 'config': config,
                    'python': sys.version, 'fixed_sha256': hashes}
        write_json(output / 'protocol.json', protocol)
        history, best_score, best_run = [], None, None
        first = 0
    instructions = (worktree / 'program.md').read_text(encoding='utf-8')
    execution = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    execution_path = output / 'executions' / (execution + '.json')
    execution_path.parent.mkdir(exist_ok=True)
    details = {'execution': execution, 'resume': resume, 'provider': provider, 'model': model,
               'experiments': experiments, 'timeout': timeout, 'rpm': rpm, 'status': 'running'}
    write_json(execution_path, details)
    summary = {'branch': branch, 'worktree': str(worktree), 'best_commit': best_commit,
               'best_log_loss': best_score, 'best_run': best_run, 'attempts': max(0, first - 1)}
    write_json(output / 'summary.json', summary)
    stop = False
    for number in range(first, first + experiments + (0 if resume else 1)):
        started = time.monotonic()
        parent = best_commit
        entry = {'experiment': number,
                 'hypothesis': 'Constant and logistic baselines' if not number else 'Proposal unavailable',
                 'parent_commit': parent, 'commit': None, 'status': 'failed', 'metrics': None,
                 'execution': execution, 'provider': provider, 'model': model}
        stop = False
        write_json(output / 'pending.json', entry)
        try:
            if number:
                context = {'instructions': instructions, 'best_score': best_score,
                           'data': {'columns': manifest.get('features', []),
                                    'history_fields': manifest.get('history_fields', []),
                                    'splits': {name: manifest.get('splits', {}).get(name)
                                               for name in ('train', 'validation')}},
                           'files': {name: (worktree / name).read_text(encoding='utf-8') for name in EDITABLE},
                           'history': history}
                proposal = proposer(context)
                if isinstance(proposal, dict) and isinstance(proposal.get('hypothesis'), str):
                    entry['hypothesis'] = proposal['hypothesis']
                (output / f'proposal_{number:03d}.json').write_text(json.dumps(proposal, indent=2), encoding='utf-8')
                write_json(output / 'pending.json', entry)
                apply_proposal(worktree, proposal)
                git(worktree, 'add', '--', *EDITABLE)
                git(worktree, 'commit', '-m', 'research: ' + proposal['hypothesis'].splitlines()[0][:120])
            entry['commit'] = git(worktree, 'rev-parse', 'HEAD')
            write_json(output / 'pending.json', entry)
            values = train(worktree, config, output / f'run_{number:03d}', timeout)
            if fingerprints(fixed) != hashes:
                raise RuntimeError('A protected file or frozen dataset changed; stopping')
            changes = git(worktree, 'status', '--porcelain', '--untracked-files=all')
            if changes or git(worktree, 'rev-parse', 'HEAD') != entry['commit']:
                raise RuntimeError('Training changed workspace files; stopping')
            score = values['candidate']['log_loss']
            entry['metrics'] = values
            if not number or score < best_score:
                best_score, best_commit = score, entry['commit']
                best_run = str(output / f'run_{number:03d}')
                entry['status'] = 'baseline' if not number else 'keep'
            else:
                entry['status'] = 'discard'
        except (Exception, KeyboardInterrupt) as error:
            entry['error'] = str(error)
            entry['status'] = 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'failed'
            if isinstance(error, KeyboardInterrupt):
                entry['status'] = 'interrupted'
            elif isinstance(error, DailyQuotaExceeded):
                entry['status'] = 'quota_exhausted'
            stop = best_score is None or isinstance(error, (DailyQuotaExceeded, KeyboardInterrupt))
        finally:
            if fingerprints(fixed) != hashes or git(worktree, 'status', '--porcelain', '--untracked-files=all'):
                # Allowed uncommitted proposal edits are recoverable; other writes stop the loop.
                changed = git(worktree, 'diff', '--name-only', 'HEAD').splitlines()
                untracked = git(worktree, 'ls-files', '--others', '--exclude-standard')
                if fingerprints(fixed) != hashes or set(changed) - set(EDITABLE) or untracked:
                    entry['error'] = 'Protected inputs or workspace changed; stopping for inspection'
                    entry['status'] = 'failed'
                    stop = True
            # Reset only this loop-owned worktree. Preserve the candidate patch in its commit.
            if git(worktree, 'rev-parse', 'HEAD') != best_commit or git(worktree, 'status', '--porcelain'):
                git(worktree, 'reset', '--hard', best_commit)
            if fingerprints(fixed) != hashes:
                entry['error'] = 'Protected inputs changed; stopping for inspection'
                stop = True
            entry['duration_seconds'] = time.monotonic() - started
            entry['timestamp'] = datetime.now(timezone.utc).isoformat()
            history.append(entry)
            save_history(output, history)
            summary = {'branch': branch, 'worktree': str(worktree), 'best_commit': best_commit,
                       'best_log_loss': best_score, 'best_run': best_run, 'attempts': number}
            write_json(output / 'summary.json', summary)
            (output / 'pending.json').unlink(missing_ok=True)
            print(f'{number}: {entry["status"]} | best log loss: {best_score}', flush=True)
            if entry.get('error'):
                print(f'  Error: {entry["error"]}', flush=True)
        if stop:
            break
    details['status'] = history[-1]['status'] if stop else 'completed'
    details['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(execution_path, details)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', help='Existing run ID to continue on this machine')
    parser.add_argument('--experiments', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--provider', choices=DEFAULT_MODELS, default='groq')
    parser.add_argument('--model', help='Model ID; defaults to openai/gpt-oss-120b (Groq) or gemini-3.8-flash (Gemini)')
    parser.add_argument('--rpm', type=int, help='Maximum API requests per minute, including retries; evenly spaced')
    parser.add_argument('--proposals', type=Path, help='JSON list of saved proposals; no API calls')
    args = parser.parse_args()
    if args.experiments < 0 or args.timeout < 1:
        parser.error('Experiments must be >= 0 and timeout must be > 0')
    if args.rpm is not None and args.rpm < 1:
        parser.error('--rpm must be a positive integer')
    if args.proposals:
        saved = json.loads(args.proposals.read_text(encoding='utf-8'))
        if not isinstance(saved, list) or len(saved) < args.experiments:
            parser.error('--proposals must contain at least --experiments proposals')
        proposals = iter(saved)
        proposer = lambda context: next(proposals)
    else:
        key_name = API_KEYS[args.provider]
        if args.experiments and not os.environ.get(key_name):
            parser.error(f'Set {key_name} or pass --proposals')
        limiter = RequestLimiter(args.rpm) if args.rpm is not None else None
        options = {'limiter': limiter} if limiter is not None else {}
        proposer = lambda context: propose(context, args.model, args.provider, **options)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume and (Path(args.resume).name != args.resume or args.resume in ('.', '..')):
        parser.error('--resume must be a run ID, not a path')
    output = ROOT / 'experiments/autoresearch' / (args.resume or stamp)
    summary = run(ROOT, output, proposer, args.experiments, args.timeout,
                  resume=bool(args.resume), provider='saved' if args.proposals else args.provider,
                  model=None if args.proposals else args.model or DEFAULT_MODELS[args.provider], rpm=args.rpm)
    print(json.dumps(summary, indent=2))
    if summary['best_log_loss'] is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
