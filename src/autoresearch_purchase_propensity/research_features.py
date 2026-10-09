"""Fixed implementation of declarative, row-wise feature operations."""
import numpy as np


def transform(frame, config):
    out = frame.copy()
    for feature in config['add']:
        name, operation, columns = feature['name'], feature['op'], feature['columns']
        if name in out.columns:
            raise ValueError(f'Cannot overwrite existing column: {name}')
        a = out[columns[0]].astype(float)
        if operation == 'log1p':
            out[name] = np.log1p(a.clip(lower=0))
        else:
            b = out[columns[1]].astype(float)
            if operation == 'ratio':
                out[name] = a / b.replace(0, np.nan)
            elif operation == 'product':
                out[name] = a * b
            elif operation == 'sum':
                out[name] = a + b
            elif operation == 'difference':
                out[name] = a - b
    return out.drop(columns=config['drop']).replace([np.inf, -np.inf], np.nan)
