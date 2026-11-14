"""Package a trusted local training artifact; never upload automatically."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact', type=Path, help='Training directory containing model.joblib and code/')
parser.add_argument('output', type=Path, help='New output .tar.gz file')
args = parser.parse_args()
artifact, output = args.artifact.resolve(), args.output.resolve()
if output.exists():
    parser.error('Output already exists; choose a new version')
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    package = root / 'code/autoresearch_purchase_propensity'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    for name in ('features.py', 'modeling.py'):
        shutil.copyfile(artifact / 'code' / name, package / name)
    for name in ('model.joblib', 'config.json', 'data_manifest.json', 'results.csv'):
        shutil.copyfile(artifact / name, root / name)
    dependencies = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'pyarrow', 'lightgbm', 'xgboost', 'catboost']
    (root / 'requirements.txt').write_text('\n'.join(f'{name}=={version(name)}' for name in dependencies) + '\n')
    manifest = {'status': 'demonstration_not_approved', 'source_artifact': str(artifact),
                'python': sys.version, 'dependencies_note': 'Packaging environment; not a reconstruction of training environment',
                'limitations': ['This packager does not assess training protocol compliance'],
                'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in root.rglob('*') if p.is_file()}}
    (root / 'package.json').write_text(json.dumps(manifest, indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
print(output)
