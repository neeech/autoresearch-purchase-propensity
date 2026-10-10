"""Freeze which earlier sessions each row can see; do not engineer aggregates."""
import pandas as pd

EXCLUDED = {'fullVisitorID', 'date', 'session_row_id', 'target',
            'transactions', 'transactionRevenue', 'history'}


def add_history(sessions):
    """Attach prior sessions from strictly earlier days, preserving input row order.

    Each history record contains the available session fields, purchased (0/1)
    and days_ago. IDs, absolute dates, revenue and current outcomes are excluded.
    Same-day sessions see the same snapshot, even across partition boundaries.
    """
    columns = [name for name in sessions.columns if name not in EXCLUDED]
    ordered = sessions.assign(_history_position=range(len(sessions))).sort_values(
        ['fullVisitorID', 'date', 'session_row_id'])
    histories = [None] * len(sessions)
    visitor, day = None, None
    previous, current, snapshot = [], [], []
    for values in ordered.itertuples(index=False, name=None):
        row = dict(zip(ordered.columns, values))
        if row['fullVisitorID'] != visitor:
            visitor, day = row['fullVisitorID'], None
            previous, current = [], []
        if row['date'] != day:
            previous.extend(current)
            current = []
            day = row['date']
            snapshot = [{**record, 'days_ago': int((day - past_day).days)}
                        for past_day, record in previous]
        histories[row['_history_position']] = snapshot
        record = {name: None if pd.isna(row[name]) else row[name] for name in columns}
        record['purchased'] = int(row['target'])
        current.append((day, record))
    result = sessions.copy()
    result['history'] = pd.Series(histories, index=result.index, dtype=object)
    return result
