"""Register a trusted packaged candidate locally; registration is not approval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('package', type=Path, help='Extracted package directory')
parser.add_argument('--selection', type=Path, default=Path('configs/candidate.json'))
parser.add_argument('--name', default='purchase-propensity')
parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
args = parser.parse_args()
root = args.package.resolve()
selection = json.loads(args.selection.read_text())
manifest = json.loads((root / 'package.json').read_text())
for name, expected in manifest['sha256'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Package integrity check failed: {name}')
sha = hashlib.sha256((root / 'model.joblib').read_bytes()).hexdigest()
if sha != selection['model_sha256']:
    raise ValueError('Package differs from selected candidate')
mlflow.set_tracking_uri(args.tracking_uri)
client = MlflowClient()
existing = client.search_model_versions(f"name = '{args.name}'")
version = next((v.version for v in existing if v.tags.get('model_sha256') == sha), None)
if version is None:
    # Load the original feature package, not editable source from today's checkout.
    sys.path.insert(0, str(root / 'code'))
    model = joblib.load(root / 'model.joblib')
    mlflow.set_experiment('purchase-propensity-models')
    with mlflow.start_run(run_name='register-' + sha[:12]):
        mlflow.set_tags({'approval': 'pending', 'source_commit': selection['commit'], 'model_sha256': sha})
        mlflow.log_metrics(selection['validation_metrics'])
        mlflow.log_artifact(str(args.selection), artifact_path='selection')
        info = mlflow.sklearn.log_model(
            model, name='model', pyfunc_predict_fn='predict_proba',
            serialization_format='cloudpickle',
            code_paths=[str(root / 'code/autoresearch_purchase_propensity')],
            pip_requirements=(root / 'requirements.txt').read_text().splitlines(),
            metadata={'approval': 'pending', 'model_sha256': sha})
        registered = mlflow.register_model(info.model_uri, args.name,
                                         tags={'approval': 'pending', 'model_sha256': sha,
                                               'source_commit': selection['commit']})
        version = registered.version
client.set_registered_model_alias(args.name, 'candidate', version)
print(json.dumps({'name': args.name, 'version': str(version),
                  'uri': f'models:/{args.name}/{version}',
                  'alias': f'models:/{args.name}@candidate', 'model_sha256': sha}))
