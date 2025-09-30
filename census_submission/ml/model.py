"""
ml/model.py — core utilities for training/evaluating a binary classifier.
"""
from __future__ import annotations
import os
from typing import Any, Dict, Iterable, Tuple
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    clf.fit(X_train, y_train.ravel() if y_train.ndim > 1 else y_train)
    return clf

def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X)

def save_model(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)

def performance_on_categorical_slice(data, feature: str, encoder, lb, model, categorical_features, label: str = "salary"):
    from ml.data import process_data
    results = {}
    for val in sorted(data[feature].dropna().unique().tolist()):
        subset = data.loc[data[feature] == val]
        if subset.empty: 
            continue
        Xs, ys, _, _ = process_data(subset, categorical_features, label, False, encoder, lb)
        preds = inference(model, Xs)
        p, r, f1 = compute_model_metrics(ys, preds)
        results[str(val)] = (p, r, f1)
    return results
