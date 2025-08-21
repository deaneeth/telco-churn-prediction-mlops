from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc

def evaluate(y_true, probs, threshold: float = 0.5) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "confusion_matrix": confusion_matrix(y_true, preds),
    }

def threshold_search(y_true, probs, metric="f1"):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    thresholds = np.append(thresholds, 1.0)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.nanargmax(f1_scores)
    return thresholds[idx], f1_scores[idx]

def compute_business_cost(y_true, probs, threshold, cost_fp: float, cost_fn: float):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"total_cost":total_cost}
