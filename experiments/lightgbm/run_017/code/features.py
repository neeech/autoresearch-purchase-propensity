"""Point-in-time history: only completed sessions on strictly earlier days."""
import numpy as np
import pandas as pd


def add_history(sessions: pd.DataFrame) -> pd.DataFrame:
    keys = ['fullVisitorID', 'date']
    daily = sessions.groupby(keys, observed=True).agg(
        sessions=('session_row_id', 'size'), purchases=('target', 'sum'),
        hits_sum=('hits', 'sum'), pageviews_sum=('pageviews', 'sum'),
        pageviews_count=('pageviews', 'count'), time_sum=('timeOnSite', 'sum'),
        time_count=('timeOnSite', 'count'),
    ).sort_index().reset_index()
    measures = ['sessions', 'purchases', 'hits_sum', 'pageviews_sum',
                'pageviews_count', 'time_sum', 'time_count']
    grouped = daily.groupby('fullVisitorID', sort=False)
    prior = grouped[measures].cumsum() - daily[measures]
    daily['history_sessions'] = prior.sessions
    daily['history_purchase_sessions'] = prior.purchases
    daily['has_history'] = prior.sessions.gt(0).astype('int8')
    daily['history_purchase_rate'] = prior.purchases / prior.sessions.replace(0, np.nan)
    daily['history_mean_hits'] = prior.hits_sum / prior.sessions.replace(0, np.nan)
    daily['history_mean_pageviews'] = prior.pageviews_sum / prior.pageviews_count.replace(0, np.nan)
    daily['history_mean_time_on_site'] = prior.time_sum / prior.time_count.replace(0, np.nan)
    daily['days_since_last_session'] = (daily.date - grouped.date.shift()).dt.days
    purchase_day = daily.date.where(daily.purchases.gt(0))
    last_purchase = purchase_day.groupby(daily.fullVisitorID).ffill()
    last_purchase = last_purchase.groupby(daily.fullVisitorID).shift()
    daily['days_since_last_purchase'] = (daily.date - last_purchase).dt.days
    cols = keys + [c for c in daily if c.startswith('history_') or c.startswith('days_since_') or c == 'has_history']
    result = sessions.merge(daily[cols], on=keys, how='left', validate='many_to_one')
    result['day_of_week'] = result.date.dt.dayofweek.astype('int8')
    result['month'] = result.date.dt.month.astype('int8')
    result['is_weekend'] = result.day_of_week.ge(5).astype('int8')
    return result
