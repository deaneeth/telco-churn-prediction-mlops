import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class Binner(BaseEstimator, TransformerMixin):
    def __init__(self, col: str, bins, labels=None):
        self.col = col
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[f"{self.col}_binned"] = pd.cut(X[self.col], bins=self.bins, labels=self.labels, include_lowest=True)
        return X
