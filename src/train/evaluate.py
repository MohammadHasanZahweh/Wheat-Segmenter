from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_classifier_logits(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_pred = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

