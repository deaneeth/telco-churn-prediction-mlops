from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

def build_model(model_name: str, config: Dict[str, Any]):
    if model_name == "random_forest":
        p = config['model']['rf']
        return RandomForestClassifier(n_estimators=p['n_estimators'], max_depth=p.get('max_depth'), min_samples_split=p.get('min_samples_split', 2), class_weight='balanced', random_state=config['seed'], n_jobs=-1)
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        p = config['model']['xgb']
        return XGBClassifier(n_estimators=p['n_estimators'], learning_rate=p['learning_rate'], use_label_encoder=False, eval_metric='logloss', random_state=config['seed'], n_jobs=-1)
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost not installed")
        p = config['model']['catboost']
        return CatBoostClassifier(iterations=p['iterations'], random_state=config['seed'], auto_class_weights='Balanced', verbose=0)
    raise ValueError("Unknown model_name")
