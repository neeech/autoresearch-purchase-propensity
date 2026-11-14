"""Verify a running inference server against trusted, ordered local predictions."""
import argparse
import csv
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8080')
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--expected', type=Path, required=True, help='Ordered CSV containing probability')
args = parser.parse_args()


def request(path, data=None, content_type='application/jsonlines'):
    call = Request(args.url.rstrip('/') + path, data=data,
                   headers={'Content-Type': content_type})
    try:
        with urlopen(call, timeout=60) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


assert request('/ping') == (200, b''), 'Health check failed'
payload = args.input.read_bytes()
expected = [float(row['probability']) for row in csv.DictReader(args.expected.open())]
assert expected and len(payload.splitlines()) == len(expected), 'Input/reference rows differ'
status, body = request('/invocations', payload)
assert status == 200, f'Prediction returned HTTP {status}'
actual = [json.loads(line)['probability'] for line in body.splitlines()]
assert len(actual) == len(expected), 'Output rows differ'
assert all(math.isfinite(p) and 0 <= p <= 1 for p in actual), 'Invalid probabilities'
assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected)), 'Predictions differ'
assert request('/invocations', b'{}')[0] == 400, 'Invalid schema was accepted'
assert request('/invocations', payload, 'text/csv')[0] == 415, 'Invalid content type was accepted'
print(json.dumps({'status': 'passed', 'rows': len(actual), 'absolute_tolerance': 1e-12,
                  'max_absolute_difference': max(abs(a-b) for a,b in zip(actual,expected)),
                  'ping': 200, 'invalid_schema': 400, 'invalid_content_type': 415}))
