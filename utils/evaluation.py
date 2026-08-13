"""Evaluation helpers shared by training, testing, and the Streamlit app."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS: tuple[str, ...] = (
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
)


def positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return scores for positive class 1 from a fitted classifier."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("The classifier did not return two-class probabilities.")
        return np.asarray(probabilities[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)

    return np.asarray(model.predict(features), dtype=float)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate the six metrics required by the assignment."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    score_values = np.asarray(y_score, dtype=float)

    auc_value = float("nan")
    if np.unique(true_values).size == 2:
        auc_value = float(roc_auc_score(true_values, score_values))

    return {
        "Accuracy": float(accuracy_score(true_values, predicted_values)),
        "AUC": auc_value,
        "Precision": float(
            precision_score(true_values, predicted_values, pos_label=1, zero_division=0)
        ),
        "Recall": float(
            recall_score(true_values, predicted_values, pos_label=1, zero_division=0)
        ),
        "F1": float(f1_score(true_values, predicted_values, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(true_values, predicted_values)),
    }


def evaluate_classifier(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Run predictions and return metrics, predictions, and positive scores."""
    predictions = np.asarray(model.predict(features), dtype=int)
    scores = positive_class_scores(model, features)
    metrics = calculate_metrics(target, predictions, scores)
    return metrics, predictions, scores


def classification_report_frame(
    target: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return a clean classification report DataFrame."""
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["Benign (0)", "Malignant (1)"],
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose()
    numeric_columns = ["precision", "recall", "f1-score", "support"]
    return frame.loc[:, numeric_columns]


def confusion_values(
    target: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return a 2x2 confusion matrix with a stable label order."""
    return confusion_matrix(target, predictions, labels=[0, 1])
