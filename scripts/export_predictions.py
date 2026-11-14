"""Predict a fixed temporal partition with trusted original model code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact', type=Path)
parser.add_argument('--split', choices=['validation', 'test'], default='validation')
parser.add_argument('--data', type=Path, default=Path('data/processed'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.artifact / 'data_manifest.json').read_text())
if manifest != json.loads((args.data / 'manifest.json').read_text()):
    parser.error('Current data manifest differs from the training artifact')
x = pd.read_parquet(args.data / f'{args.split}_X.parquet').drop(columns='month')
y = pd.read_parquet(args.data / f'{args.split}_y.parquet')
meta = pd.read_parquet(args.data / f'{args.split}_metadata.parquet')
if len(x) != len(y) or len(x) != len(meta) or not y.session_row_id.equals(meta.session_row_id):
    parser.error('Partition rows are not aligned')
if 'history' in x:
    x['history'] = x.history.map(lambda events: [] if events is None else list(events))
# Fresh process + original package: joblib must not import today's editable features.
with tempfile.TemporaryDirectory() as temporary:
    shutil.copytree(args.artifact / 'code', Path(temporary) / 'autoresearch_purchase_propensity')
    sys.path.insert(0, temporary)
    model = joblib.load(args.artifact / 'model.joblib')
    probability = model.predict_proba(x)[:, 1]
constant = json.loads((args.artifact / 'constant.json').read_text())['probability']
args.output.mkdir(parents=True, exist_ok=False)
for name, values in [('candidate', probability), ('constant', np.full(len(y), constant))]:
    pd.DataFrame({'session_row_id': y.session_row_id, 'target': y.target,
                  'probability': values}).to_csv(args.output / f'{name}.csv', index=False)
(args.output / 'source.json').write_text(json.dumps({
    'artifact': str(args.artifact.resolve()),
    'model_sha256': hashlib.sha256((args.artifact / 'model.joblib').read_bytes()).hexdigest(),
    'split': args.split,
    f'{args.split}_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.data.glob(f'{args.split}_*.parquet'))},
    'rows': len(y), 'retrained': False,
    'effective_params': model.named_steps['model'].get_params(deep=False),
}, indent=2, default=str))
print(f'Exported {len(y)} {args.split} rows to {args.output}')
