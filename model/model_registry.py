"""Shared model names and artifact locations for the project."""

from __future__ import annotations

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "Logistic Regression": {
        "artifact": "logistic_regression.joblib",
        "family": "Linear classifier",
        "summary": "A regularized linear baseline trained on standardized features.",
    },
    "Decision Tree": {
        "artifact": "decision_tree.joblib",
        "family": "Tree classifier",
        "summary": "A depth-controlled decision tree that captures nonlinear rules.",
    },
    "K-Nearest Neighbors": {
        "artifact": "knn.joblib",
        "family": "Instance-based classifier",
        "summary": "A distance-based classifier trained on standardized features.",
    },
    "Gaussian Naive Bayes": {
        "artifact": "gaussian_naive_bayes.joblib",
        "family": "Probabilistic classifier",
        "summary": "A Gaussian conditional-independence model with fast inference.",
    },
    "Random Forest": {
        "artifact": "random_forest.joblib",
        "family": "Ensemble classifier",
        "summary": "An ensemble of randomized decision trees with balanced class weights.",
    },
    "Support Vector Machine": {
        "artifact": "support_vector_machine.joblib",
        "family": "Kernel classifier",
        "summary": "An RBF-kernel classifier included as the sixth model requested by the brief.",
    },
}

MODEL_DISPLAY_ORDER: tuple[str, ...] = tuple(MODEL_REGISTRY.keys())
