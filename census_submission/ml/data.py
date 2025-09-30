"""
ml/data.py — processing helpers (fit/transform) compatible with the rubric.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def process_data(
    X: pd.DataFrame,
    categorical_features: List[str],
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """
    Fits or applies a OneHotEncoder on categorical features and binarizes the label.
    Returns (X_processed, y, encoder, label_binarizer). If label is None, y will be an empty array.
    """
    X = X.copy()
    y = None
    if label is not None and label in X.columns:
        y = X[label]
        X = X.drop(columns=[label])

    X_cat = X[categorical_features].astype(str)
    X_num = X.drop(columns=categorical_features)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_ohe = encoder.fit_transform(X_cat)
        lb = LabelBinarizer()
        if y is not None:
            y = lb.fit_transform(y.values).ravel()
        else:
            y = np.array([])
    else:
        if encoder is None or lb is None:
            raise ValueError("Must provide fitted encoder and label binarizer when training=False")
        X_cat_ohe = encoder.transform(X_cat)
        if y is not None:
            y = lb.transform(y.values).ravel()
        else:
            y = np.array([])

    Xp = np.hstack([X_cat_ohe, X_num.values])
    return Xp, y, encoder, lb
