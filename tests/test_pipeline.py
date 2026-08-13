"""Smoke tests for the saved datasets, models, and required metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from model.model_registry import MODEL_DISPLAY_ORDER, MODEL_REGISTRY
from utils.evaluation import METRIC_COLUMNS, evaluate_classifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "model" / "artifacts"


def test_dataset_meets_assignment_minimums() -> None:
    full_data = pd.read_csv(ROOT / "data" / "breast_cancer_wisconsin_full.csv")
    assert full_data.shape[0] >= 500
    assert full_data.shape[1] - 1 >= 12
    assert "target" in full_data.columns


def test_all_six_saved_models_evaluate() -> None:
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    test_data = pd.read_csv(ROOT / "test_data.csv")
    features = test_data.loc[:, metadata["feature_names"]]
    target = test_data["target"]

    assert len(MODEL_DISPLAY_ORDER) == 6
    for model_name in MODEL_DISPLAY_ORDER:
        model_path = ARTIFACT_DIR / MODEL_REGISTRY[model_name]["artifact"]
        assert model_path.exists(), f"Missing artifact for {model_name}"
        model = joblib.load(model_path)
        metrics, predictions, scores = evaluate_classifier(model, features, target)
        assert len(predictions) == len(test_data)
        assert len(scores) == len(test_data)
        assert set(METRIC_COLUMNS).issubset(metrics)
        assert 0.0 <= metrics["Accuracy"] <= 1.0
        assert 0.0 <= metrics["Precision"] <= 1.0
        assert 0.0 <= metrics["Recall"] <= 1.0
        assert 0.0 <= metrics["F1"] <= 1.0
        assert -1.0 <= metrics["MCC"] <= 1.0


def test_precomputed_metric_table_has_required_columns() -> None:
    metrics = pd.read_csv(ARTIFACT_DIR / "metrics.csv")
    assert metrics.shape[0] == 6
    assert ["Model", *METRIC_COLUMNS] == list(metrics.columns)
