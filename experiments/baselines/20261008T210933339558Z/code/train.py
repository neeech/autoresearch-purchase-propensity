"""Reproducible baseline suite. Train on January-April, evaluate May only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from .evaluate import evaluate_segments, calibration_table

from .modeling import FORBIDDEN, feature_columns, build_model


def load_partition(directory, name, expected_features):
    if name not in ['train', 'validation']:
        raise ValueError('Baseline runner only permits train and validation')
    x = pd.read_parquet(directory / f'{name}_X.parquet')
    y = pd.read_parquet(directory / f'{name}_y.parquet')
    metadata = pd.read_parquet(directory / f'{name}_metadata.parquet')
    if len(x) != len(y) or len(y) != len(metadata) or not y.session_row_id.equals(metadata.session_row_id):
        raise ValueError('Partition files do not align')
    if x.columns.tolist() != expected_features or FORBIDDEN.intersection(x.columns):
        raise ValueError('Unexpected or unsafe feature schema')
    if not y.target.isin([0, 1]).all() or y.session_row_id.duplicated().any():
        raise ValueError('Invalid labels or duplicate row identifiers')
    return x, y.target.to_numpy(), metadata


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config):
    if config['primary_metric'] != 'log_loss' or config['direction'] != 'minimize':
        raise ValueError('Baseline protocol fixes log loss minimization')
    directory = Path(config['data_dir'])
    manifest = json.loads((directory / 'manifest.json').read_text())
    train_x, train_y, train_meta = load_partition(directory, 'train', manifest['features'])
    val_x, val_y, val_meta = load_partition(directory, 'validation', manifest['features'])
    if train_meta.date.max() >= val_meta.date.min() or set(train_meta.session_row_id) & set(val_meta.session_row_id):
        raise ValueError('Temporal partitions must be disjoint and chronological')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Path(config['output_root']) / run_id
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / 'code'
    shutil.copytree(Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    versions = {'python_note': 'see environment.txt', 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}
    import sys
    (root / 'environment.txt').write_text(sys.version + '\n' + json.dumps(versions, indent=2))
    inputs = {f'{name}_{suffix}.parquet': file_hash(directory / f'{name}_{suffix}.parquet')
              for name in ['train', 'validation'] for suffix in ['X', 'y', 'metadata']}
    protocol = {'run_id': run_id, 'config': config, 'versions': versions,
                'data_manifest': manifest, 'input_sha256': inputs,
                'code_sha256': {str(p.relative_to(snapshot)): file_hash(p) for p in snapshot.rglob('*.py')},
                'evaluation': 'train only on train; compare only validation; test not read',
                'calendar_policy': 'day_of_week categorical and is_weekend; month excluded to avoid extrapolating an unseen month',
                'class_weight': None, 'hypotheses': {
                    'constant': 'Reference probability estimated from train prevalence.',
                    'context_only': 'Acquisition, device and visitor context provide signal without session activity.',
                    'session': 'Completed-session activity improves classification over context.',
                    'session_history': 'Strictly prior-day user history adds signal to session features.',
                    'session_history_no_city': 'Assess dependence on city availability in the demo extract.'}}
    # Persist protocol and exact code before fitting any model.
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    train_visitors = set(train_meta.fullVisitorID)
    all_results, rows = {}, []
    for variant in config['variants']:
        folder = root / variant
        folder.mkdir()
        start = time.monotonic()
        columns = [] if variant == 'constant' else feature_columns(variant, train_x.columns)
        result = {'variant': variant, 'features': columns, 'status': 'running'}
        (folder / 'config.json').write_text(json.dumps({'hypothesis': protocol['hypotheses'][variant], 'features': columns, 'config': config}, indent=2))
        try:
            if variant == 'constant':
                model = {'type': 'constant', 'probability': float(train_y.mean())}
                probabilities = np.full(len(val_y), model['probability'])
                caught = []
            else:
                model = build_model(columns, config)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(train_x[columns], train_y)
                probabilities = model.predict_proba(val_x[columns])[:, 1]
            result['warnings'] = [str(w.message) for w in caught]
            result['converged'] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            result['metrics'] = evaluate_segments(val_y, probabilities, val_meta, train_visitors)
            result['elapsed_seconds'] = time.monotonic() - start
            result['status'] = 'ok' if result['converged'] else 'nonconverged'
            if variant != 'constant':
                result['iterations'] = model.named_steps['model'].n_iter_.tolist()
                result['encoded_features'] = len(model.named_steps['features'].get_feature_names_out())
            joblib.dump(model, folder / 'model.joblib')
            pd.DataFrame({'session_row_id': val_meta.session_row_id, 'target': val_y, 'probability': probabilities}).to_parquet(folder / 'validation_predictions.parquet', index=False)
            calibration_table(val_y, probabilities).to_csv(folder / 'calibration.csv', index=False)
            rows.append({'variant': variant, 'status': result['status'], **result['metrics']['all'], 'elapsed_seconds': result['elapsed_seconds']})
            print(f"{variant}: {result['status']} log_loss={result['metrics']['all']['log_loss']:.6f} AP={result['metrics']['all']['average_precision']:.6f}", flush=True)
        except Exception as error:
            result.update(status='error', error=repr(error), elapsed_seconds=time.monotonic()-start)
            print(f'{variant}: error {error}', flush=True)
        (folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        all_results[variant] = result
        (root / 'results.json').write_text(json.dumps(all_results, indent=2, allow_nan=False))
    valid = [r for r in all_results.values() if r['status'] == 'ok']
    if not valid:
        raise RuntimeError(f'No successful experiments; see {root}')
    winner = min(valid, key=lambda r: r['metrics']['all']['log_loss'])
    summary = pd.DataFrame(rows).sort_values('log_loss')
    summary.to_csv(root / 'leaderboard.csv', index=False)
    best = {'run_id': run_id, 'variant': winner['variant'], 'validation_log_loss': winner['metrics']['all']['log_loss'],
            'model_path': str(root / winner['variant'] / 'model.joblib'), 'test_evaluated': False}
    (root / 'best.json').write_text(json.dumps(best, indent=2))
    make_report(root, all_results, summary, best)
    if any(r['status'] != 'ok' for r in all_results.values()):
        raise RuntimeError(f'Incomplete suite; inspect {root}')
    (Path(config['output_root']) / 'latest.json').write_text(json.dumps({'run_dir': str(root), **best}, indent=2))
    print(f'Results: {root}', flush=True)
    return root


def make_report(root, results, summary, best):
    from .eda import plt
    from .eda_detail import Report
    out = Path('reports')
    r = Report()
    r.heading('Baselines: evaluación en mayo', 1)
    r.paragraph('Entrenamiento enero–abril, validación mayo. Junio no se leyó ni evaluó. Todas las transformaciones aprendidas se ajustaron sólo con entrenamiento; se conserva el supuesto de transacciones vacías=0. Comparaciones iniciales, sin búsqueda de hiperparámetros.')
    r.table(summary[['variant', 'status', 'log_loss', 'average_precision', 'brier', 'roc_auc', 'mean_probability', 'elapsed_seconds']])
    r.paragraph(f'Menor log loss: {best["variant"]} ({best["validation_log_loss"]:.6f}). Average Precision mide ranking; Brier y log loss evalúan probabilidades. ROC-AUC es complementaria. No se usan pesos de clase ni remuestreo para evitar cambiar la prevalencia implícita.')
    if 'session_history' in results and 'session' in results and results['session_history']['status']=='ok' and results['session']['status']=='ok':
        delta = results['session']['metrics']['all']['log_loss'] - results['session_history']['metrics']['all']['log_loss']
        r.paragraph(f'Aporte de historia: reducción de log loss de {delta:.6f} respecto a session (positivo significa mejora). Diferencia observada en una única validación temporal; no es prueba estadística de superioridad.')
    segments=[]
    for name, result in results.items():
        if 'metrics' in result:
            for segment, values in result['metrics'].items():
                segments.append({'variant':name,'segment':segment,**values})
    segments=pd.DataFrame(segments)
    segments.to_csv(root / 'segment_metrics.csv',index=False)
    r.heading('Usuarios con y sin historia')
    r.table(segments[segments.segment.isin(['without_prior_day_history','with_prior_day_history'])][['variant','segment','rows','positives','log_loss','average_precision','brier']])
    r.paragraph('Sin historia significa sin sesiones en días anteriores observados; puede haber visitas anteriores al archivo. Las métricas adicionales por usuario visto/no visto en train están en segment_metrics.csv.')
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name, result in results.items():
        if 'metrics' not in result:
            continue
        calibration=pd.read_csv(root / name / 'calibration.csv')
        axes[0].plot(calibration.mean_probability, calibration.observed_rate, marker='o',label=name)
        predictions=pd.read_parquet(root / name / 'validation_predictions.parquet')
        from sklearn.metrics import precision_recall_curve
        precision,recall,_=precision_recall_curve(predictions.target,predictions.probability)
        axes[1].plot(recall,precision,label=name)
    axes[0].plot([0,1],[0,1],'--',color='gray',label='ideal')
    axes[0].set(xlabel='Probabilidad media predicha',ylabel='Frecuencia de compra',title='Calibración · bins de igual volumen')
    axes[0].set_xlim(0,.5); axes[0].set_ylim(0,.5)
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–Recall · validación',xlim=(0,1),ylim=(0,1))
    for ax in axes:
        ax.legend(fontsize=7)
    figure = out / 'figures' / f'baselines_{root.name}'
    for ext in ['png','pdf']:
        fig.savefig(figure.with_suffix('.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    r.image('figures/'+figure.name+'.png','Calibración y Precision–Recall')
    r.paragraph('Cada punto de calibración agrupa probabilidades similares; la tabla por modelo incluye volúmenes y extremos. Calibración mostrada hasta 0,5; no se ajustó un calibrador con esta validación. La actividad corresponde a sesiones terminadas y puede reflejar consecuencias de compra.')
    r.heading('Reproducibilidad y siguiente etapa')
    r.paragraph(f'Artefactos: {root}. Cada candidato conserva snapshot de código, configuración, hashes de inputs, versiones, modelo, probabilidades y métricas. No se descartan modelos perdedores. La regularización C=1 y demás opciones están en configs/baselines.json. month se excluye por ser un mes nuevo en validación; day_of_week se codifica como categoría.')
    r.paragraph('Siguiente etapa: usar esta referencia para experimentos acotados de modelos de árboles, ingeniería de features y regularización. Evaluar mejoras con el mismo protocolo y reservar junio para la selección final. No calibrar sobre mayo y reportar el rendimiento en las mismas filas.')
    # Report.save has fixed EDA names; write in a staging subdirectory then rename.
    temp = root / 'report'
    temp.mkdir()
    r.save(temp)
    shutil.copyfile(temp / 'eda.md', out / 'baselines.md')
    shutil.copyfile(temp / 'eda.html', out / 'baselines.html')
    shutil.rmtree(temp)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baselines.json')
    args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__=='__main__':
    main()
