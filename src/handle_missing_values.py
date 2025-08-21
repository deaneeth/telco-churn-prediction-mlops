from typing import Optional
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that imputes numeric and categorical columns.
    Useful in ColumnTransformer pipelines.
    """
    def __init__(self, numeric_strategy: str = "median", categorical_strategy: str = "most_frequent"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.num_imputer = None
        self.cat_imputer = None

    def fit(self, X: pd.DataFrame, y=None):
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
        if num_cols:
            self.num_imputer = SimpleImputer(strategy=self.numeric_strategy).fit(X[num_cols])
        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy=self.categorical_strategy, fill_value="missing").fit(X[cat_cols])
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
        if num_cols and self.num_imputer:
            X[num_cols] = self.num_imputer.transform(X[num_cols])
        if cat_cols and self.cat_imputer:
            X[cat_cols] = self.cat_imputer.transform(X[cat_cols])
        return X
