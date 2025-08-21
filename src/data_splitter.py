from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    # split trainval into train + val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
