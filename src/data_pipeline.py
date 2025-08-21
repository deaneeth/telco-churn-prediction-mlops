from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from .feature_encoding import EncoderFactory
from .feature_scaling import ScalerFactory
from sklearn.pipeline import make_pipeline

def build_preprocessing_pipeline(numeric_features: List[str], categorical_features: List[str], config: dict):
    # numeric pipeline: impute + scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=config['processing']['numeric_impute'])),
        ('scaler', ScalerFactory.get_scaler(config['processing']['scaler']))
    ])
    # categorical pipeline: impute + encode
    cat_enc = EncoderFactory.get_encoder(config['processing']['encoder'])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=config['processing']['categorical_impute'], fill_value='missing')),
        ('encoder', cat_enc)
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor
