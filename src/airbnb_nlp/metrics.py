"""Shared evaluation metrics so every model is scored the same way."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Accuracy, macro F1, per-class F1 and the confusion matrix (rows = true, cols = predicted)."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_low": round(float(f1_score(y_true, y_pred, pos_label=0)), 4),
        "f1_high": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": len(y_true),
    }
