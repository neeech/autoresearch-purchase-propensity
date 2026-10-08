"""Validated ingestion, point-in-time history and temporal partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .features import add_history

REQUIRED = ['fullVisitorID', 'visitNumber', 'date', 'bounces', 'hits', 'pageviews',
            'timeOnSite', 'transactionRevenue', 'transactions', 'source',
            'channelGrouping', 'browser', 'deviceCategory', 'country', 'city']
CATEGORICAL = ['source', 'channelGrouping', 'browser', 'deviceCategory', 'country', 'city']


def load_sessions(path, missing_transactions_mean_zero=True):
    data = pd.read_csv(path, dtype={'fullVisitorID': 'string'}, low_memory=False)
    if set(data.columns) != set(REQUIRED):
        raise ValueError('Unexpected input schema')
    if data.fullVisitorID.isna().any() or data.fullVisitorID.str.strip().eq('').any():
        raise ValueError('Missing visitor identifiers')
    data['date'] = pd.to_datetime(data.date.astype(str), format='%Y%m%d', errors='raise')
    if data.date.isna().any():
        raise ValueError('Missing dates')
    for col in ['visitNumber', 'bounces', 'hits', 'pageviews', 'timeOnSite', 'transactions', 'transactionRevenue']:
        data[col] = pd.to_numeric(data[col], errors='raise')
        if data[col].dropna().lt(0).any():
            raise ValueError(f'Negative values in {col}')
    if data[['hits', 'visitNumber']].isna().any().any():
        raise ValueError('Missing required counters')
    if not missing_transactions_mean_zero and data.transactions.isna().any():
        raise ValueError('Target missingness requires an explicit assumption')
    data['session_row_id'] = range(len(data))
    data['target'] = data.transactions.fillna(0).gt(0).astype('int8')
    for col in CATEGORICAL:
        data[col] = data[col].astype('string').replace(
            {'(not set)': pd.NA, 'not available in demo dataset': pd.NA})
    # Preserve unknown durations/bounces instead of assuming zero.
    for col in ['bounces', 'pageviews', 'timeOnSite']:
        data[f'{col}_missing'] = data[col].isna().astype('int8')
    return data


def temporal_split(data, validation_start, test_start):
    val, test = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if val >= test:
        raise ValueError('Validation must precede test')
    parts = {'train': data[data.date < val].copy(),
             'validation': data[(data.date >= val) & (data.date < test)].copy(),
             'test': data[data.date >= test].copy()}
    if any(part.empty for part in parts.values()):
        raise ValueError('Every partition must contain rows')
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    sessions = load_sessions(config['raw_path'], config['missing_transactions_mean_zero'])
    from .eda import generate_report
    # EDA precedes feature construction; test labels are descriptive only.
    generate_report(sessions, Path(config['report_dir']))
    enriched = add_history(sessions)
    parts = temporal_split(enriched, config['validation_start'], config['test_start'])
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    excluded = ['fullVisitorID', 'date', 'session_row_id', 'target', 'transactions', 'transactionRevenue']
    features = [c for c in enriched if c not in excluded]
    summary = {}
    for name, part in parts.items():
        part = part.sort_values(['date', 'session_row_id'])
        part[features].to_parquet(out / f'{name}_X.parquet', index=False)
        part[['session_row_id', 'target']].to_parquet(out / f'{name}_y.parquet', index=False)
        part[['session_row_id', 'fullVisitorID', 'date', 'has_history']].to_parquet(out / f'{name}_metadata.parquet', index=False)
        summary[name] = {'rows': len(part), 'positives': int(part.target.sum()),
                         'purchase_rate': float(part.target.mean()),
                         'with_prior_day_history_rate': float(part.has_history.mean()),
                         'start': str(part.date.min().date()), 'end': str(part.date.max().date())}
    manifest = {'config': config, 'raw_sha256': hashlib.sha256(Path(config['raw_path']).read_bytes()).hexdigest(),
                'features': features, 'excluded': excluded, 'splits': summary,
                'history_policy': 'strictly earlier days, updated sequentially across partitions',
                'target_assumption': 'missing transactions means zero',
                'versions': {'pandas': pd.__version__}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (Path(config['report_dir']) / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
