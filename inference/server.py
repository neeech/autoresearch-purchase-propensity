"""SageMaker WSGI application: JSON Lines in, probabilities in the same order out."""
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


def create_app(model_dir=None):
    root = Path(model_dir or os.environ.get('MODEL_DIR', '/opt/ml/model'))
    sys.path.insert(0, str(root / 'code'))
    model = joblib.load(root / 'model.joblib')
    columns = [c for c in json.loads((root / 'data_manifest.json').read_text())['features'] if c != 'month']
    # Raw session types are independent of model-specific preprocessing steps.
    numeric = {'visitNumber', 'bounces', 'hits', 'pageviews', 'timeOnSite', 'day_of_week'}

    def app(environ, start_response):
        status, body, content_type = '200 OK', b'', 'text/plain'
        path, method = environ.get('PATH_INFO'), environ.get('REQUEST_METHOD')
        if path == '/ping' and method == 'GET':
            pass  # Model already loaded successfully before accepting requests.
        elif path == '/invocations' and method == 'POST':
            if environ.get('CONTENT_TYPE', '').split(';')[0] not in ('application/jsonlines', 'application/x-ndjson'):
                status, body = '415 Unsupported Media Type', b'Use application/jsonlines'
            else:
                try:
                    length = int(environ.get('CONTENT_LENGTH') or 0)
                    if not 0 < length <= 6 * 1024 * 1024:
                        raise ValueError('Payload must be between 1 byte and 6 MB')
                    records = [json.loads(line) for line in environ['wsgi.input'].read(length).splitlines() if line.strip()]
                    if not records or any(not isinstance(row, dict) or set(row) != set(columns) for row in records):
                        raise ValueError('Every row must contain exactly the model input columns')
                    frame = pd.DataFrame(records, columns=columns)
                    for column in set(numeric).intersection(columns):
                        frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
                    if 'history' in frame and not frame.history.map(lambda events: isinstance(events, list)).all():
                        raise ValueError('history must be a list, including [] for new users')
                    probabilities = model.predict_proba(frame)[:, 1]
                    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                        raise RuntimeError('Invalid model probabilities')
                    body = ''.join(json.dumps({'probability': float(p)}) + '\n' for p in probabilities).encode()
                    content_type = 'application/jsonlines'
                except (ValueError, TypeError, KeyError):
                    status, body = '400 Bad Request', b'Invalid input; check JSON, columns, types and history'
                except Exception:
                    status, body = '500 Internal Server Error', b'Prediction failed'
        else:
            status, body = '404 Not Found', b'Not found'
        start_response(status, [('Content-Type', content_type), ('Content-Length', str(len(body)))])
        return [body]

    return app


if __name__ == '__main__':
    os.execvp('gunicorn', ['gunicorn', '--bind', '0.0.0.0:8080', '--workers', '1',
                          '--threads', '1', '--timeout', '60', '--access-logfile', '-', 'server:create_app()'])
