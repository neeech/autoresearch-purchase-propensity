"""Importable feature transforms and baseline estimators, including artifact reloads."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CONTEXT = ['visitNumber', 'source', 'channelGrouping', 'browser', 'deviceCategory',
           'country', 'city', 'day_of_week', 'is_weekend']
ACTIVITY = ['bounces', 'hits', 'pageviews', 'timeOnSite', 'pageviews_missing', 'timeOnSite_missing']
CATEGORIES = ['source', 'channelGrouping', 'browser', 'deviceCategory', 'country', 'city', 'day_of_week']
LOG_COLUMNS = ['visitNumber', 'hits', 'pageviews', 'timeOnSite', 'history_sessions',
               'history_purchase_sessions', 'history_mean_hits', 'history_mean_pageviews',
               'history_mean_time_on_site', 'days_since_last_session', 'days_since_last_purchase']
FORBIDDEN = {'transactions', 'transactionRevenue', 'target', 'fullVisitorID', 'date', 'session_row_id'}


def feature_columns(variant, available):
    history = [c for c in available if c.startswith('history_') or c.startswith('days_since_') or c == 'has_history']
    choices = {'context_only': CONTEXT, 'session': CONTEXT + ACTIVITY,
               'session_history': CONTEXT + ACTIVITY + history,
               'session_history_no_city': [c for c in CONTEXT + ACTIVITY + history if c != 'city']}
    if variant not in choices:
        raise ValueError(f'Unknown model variant: {variant}')
    columns = choices[variant]
    if FORBIDDEN.intersection(columns) or not set(columns).issubset(available):
        raise ValueError('Unsafe or missing features')
    return columns


def prepare_input(frame):
    """Fixed transforms, with imputation/encoding fitted later only on train."""
    out = frame.copy()
    for c in out:
        if c in CATEGORIES:
            # Explicit object/np.nan avoids pd.NA ambiguity in sklearn imputers.
            values = out[c].astype('string')
            out[c] = values.astype(object).where(values.notna(), np.nan)
        elif c in LOG_COLUMNS:
            out[c] = np.log1p(out[c].astype(float))
    return out


def build_model(columns, config):
    categorical = [c for c in columns if c in CATEGORIES]
    numeric = [c for c in columns if c not in CATEGORIES]
    encoder = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
                        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                min_frequency=config['min_category_frequency']))])
    numbers = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                        ('scale', StandardScaler())])
    transform = ColumnTransformer([('categorical', encoder, categorical), ('numeric', numbers, numeric)], sparse_threshold=1.0)
    return Pipeline([('prepare', FunctionTransformer(prepare_input, validate=False, feature_names_out='one-to-one')),
                     ('features', transform),
                     ('model', LogisticRegression(**config['logistic'], random_state=config['seed']))])


