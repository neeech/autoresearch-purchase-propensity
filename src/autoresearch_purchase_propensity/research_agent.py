"""Minimal Groq client: configuration proposals, without tools or commands."""
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROMPT = '''Propose ONE hypothesis to reduce validation log loss for purchase propensity.
Return JSON: {"hypothesis":"...","reasoning":"...","changes":[{"file":"...","old":"...","new":"..."}]}.
Only modify the supplied JSON files, using exact unique old fragments.
Feature schema: {"add":[{"name":"hits_per_page","op":"ratio","columns":["hits","pageviews"]}],"drop":[]}.
Operations: log1p (one column), ratio/product/sum/difference (two columns).
Use available columns, preserve rows, never overwrite existing columns.
No code, commands, imports, labels or statistics learned on validation.
Use the supplied model parameters. Do not repeat previously tested hypotheses.'''


def reject_constant(value):
    raise ValueError(f'Invalid JSON number: {value}')


class ResearchAgent:
    def __init__(self, model='openai/gpt-oss-120b'):
        self.model = model
        self.key = os.environ.get('GROQ_API_KEY')
        if not self.key:
            raise ValueError('Set GROQ_API_KEY with a key from https://console.groq.com/keys')

    def propose(self, context):
        payload = {'model': self.model, 'temperature': 0.2, 'max_completion_tokens': 4096,
                   'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'system', 'content': PROMPT},
                                {'role': 'user', 'content': json.dumps(context)}]}
        content = self.request('chat/completions', payload)['choices'][0]['message']['content']
        return json.loads(content, parse_constant=reject_constant)

    def request(self, path, payload=None):
        request = Request('https://api.groq.com/openai/v1/' + path,
                          json.dumps(payload).encode() if payload is not None else None,
                          {'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json',
                           'User-Agent': 'purchase-autoresearch/0.1'})
        try:
            with urlopen(request, timeout=180) as response:
                return json.load(response)
        except HTTPError as error:
            body = error.read(4096).decode(errors='replace')
            try:
                message = json.loads(body)['error']['message']
            except (ValueError, KeyError, TypeError):
                message = body[:300]
            error.msg = f'{error.reason}: {str(message).replace(self.key, "[REDACTED]")}'
            raise

    def check(self):
        models = self.request('models')['data']
        if self.model not in {model['id'] for model in models}:
            raise ValueError(f'Model not available: {self.model}')
