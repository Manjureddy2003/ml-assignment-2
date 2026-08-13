"""Tests for uploaded CSV validation and target conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from utils.data_validation import prepare_uploaded_frame

ROOT = Path(__file__).resolve().parents[1]


def feature_names() -> list[str]:
    metadata = json.loads(
        (ROOT / "model" / "artifacts" / "metadata.json").read_text()
    )
    return list(metadata["feature_names"])


def test_bundled_test_data_is_valid() -> None:
    frame = pd.read_csv(ROOT / "test_data.csv")
    prepared = prepare_uploaded_frame(frame, feature_names())
    assert prepared.features.shape == (114, 30)
    assert prepared.target is not None
    assert set(prepared.target.unique()) == {0, 1}


def test_text_labels_are_supported() -> None:
    frame = pd.read_csv(ROOT / "test_data.csv").head(4)
    frame["diagnosis"] = frame["target"].map({0: "B", 1: "M"})
    frame = frame.drop(columns=["target"])
    prepared = prepare_uploaded_frame(frame, feature_names())
    assert prepared.target is not None
    assert prepared.target.tolist() == [0 if value == "B" else 1 for value in frame["diagnosis"]]


def test_missing_feature_is_rejected() -> None:
    frame = pd.read_csv(ROOT / "test_data.csv").drop(columns=[feature_names()[0]])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_uploaded_frame(frame, feature_names())
