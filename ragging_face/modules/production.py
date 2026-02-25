import pandas as pd
import numpy as np


def analyze_csv(file):
    """Read production log CSV and compute KPIs and correlations."""
    df = pd.read_csv(file)
    # ensure numeric
    numeric = df.select_dtypes(include=[np.number])
    kpis = {}
    if 'defect' in df.columns:
        # assume binary or count
        defect_rate = df['defect'].mean()
        kpis['defect_rate'] = defect_rate
    else:
        kpis['defect_rate'] = None
    # compute basic stats for numeric columns
    stats = numeric.describe().to_dict()
    # find correlations with defect if present
    corr = None
    if 'defect' in df.columns:
        corr = numeric.corr()['defect'].sort_values(ascending=False).to_dict()
    return {'kpis': kpis, 'stats': stats, 'correlation': corr, 'df': df}


def detect_anomalies(df, column, z_thresh=3):
    """Return indices of anomalies in a numeric column using z-score"""
    if column not in df:
        return []
    series = df[column]
    mean = series.mean()
    std = series.std()
    z = (series - mean) / (std if std != 0 else 1)
    return df.index[np.abs(z) > z_thresh].tolist()
