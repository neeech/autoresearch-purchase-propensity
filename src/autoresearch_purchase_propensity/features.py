"""Editable session features and history aggregations; availability is fixed."""
import numpy as np

CATEGORIES = ['source', 'channelGrouping', 'browser', 'deviceCategory', 'country', 'city', 'day_of_week']
LOG_COLUMNS = ['visitNumber', 'hits', 'pageviews', 'timeOnSite']


def build_features(frame):
    """Preserve rows; learned transforms belong in the train-fitted pipeline."""
    out = frame.copy()
    # Initial baseline: current session only. Experiments may aggregate frame.history
    # before dropping it, e.g. counts, purchase rates, recency or 7/30-day windows.
    out = out.drop(columns='history', errors='ignore')
    if 'day_of_week' in out:
        out['is_weekend'] = out.day_of_week.ge(5).astype('int8')
    for name in ('pageviews', 'timeOnSite'):
        if name in out:
            out[f'{name}_missing'] = out[name].isna().astype('int8')
    for name in out:
        if name in CATEGORIES:
            values = out[name].astype('string')
            out[name] = values.astype(object).where(values.notna(), np.nan)
        elif name in LOG_COLUMNS:
            out[name] = np.log1p(out[name].astype(float))
    return out
