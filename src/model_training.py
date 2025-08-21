from typing import Any, Dict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from joblib import dump
from .utils import get_logger, ensure_dir

logger = get_logger("model_training")

class Trainer:
    def __init__(self, preprocessor, model, config: Dict[str, Any], artifacts_dir: str):
        self.preprocessor = preprocessor
        self.model = model
        self.config = config
        self.artifacts_dir = artifacts_dir

    def build_pipeline(self):
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('clf', self.model)
        ])
        return pipeline

    def fit(self, X, y, param_distributions=None, n_iter=20):
        pipeline = self.build_pipeline()
        cv = StratifiedKFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=self.config['seed'])
        if param_distributions:
            search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_iter, scoring=self.config['training']['scoring'], cv=cv, n_jobs=-1, verbose=2, random_state=self.config['seed'])
            search.fit(X, y)
            best = search.best_estimator_
            logger.info(f"RandomizedSearch best score: {search.best_score_}")
            ensure_dir(self.artifacts_dir)
            dump(search, f"{self.artifacts_dir}/search_{type(self.model).__name__}.joblib")
            return best, search
        else:
            pipeline.fit(X, y)
            ensure_dir(self.artifacts_dir)
            dump(pipeline, f"{self.artifacts_dir}/{type(self.model).__name__}_pipeline.joblib")
            return pipeline, None
