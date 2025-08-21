from typing import Tuple, List
import numpy as np
import pandas as pd

def iqr_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

def zscore_outliers(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    z = (series - series.mean()) / (series.std() + 1e-9)
    return z.abs() > z_thresh

def remove_outliers(df: pd.DataFrame, cols: List[str], method: str = "iqr") -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for c in cols:
        if method == "iqr":
            mask = mask | iqr_outliers(df[c])
        else:
            mask = mask | zscore_outliers(df[c])
    return df.loc[~mask].reset_index(drop=True)
