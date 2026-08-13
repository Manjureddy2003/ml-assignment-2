"""CSV normalization and validation for uploaded evaluation data."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PreparedDataset:
    """Validated data ready for model evaluation or prediction."""

    features: pd.DataFrame
    target: pd.Series | None
    normalized_frame: pd.DataFrame
    target_column: str | None
    ignored_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_column_name(value: object) -> str:
    """Convert a column name into lowercase snake_case."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _map_target_value(value: object) -> int | float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        integer_value = int(value)
        if integer_value in (0, 1):
            return integer_value

    normalized = str(value).strip().lower()
    mapping = {
        "0": 0,
        "b": 0,
        "benign": 0,
        "negative": 0,
        "1": 1,
        "m": 1,
        "malignant": 1,
        "positive": 1,
    }
    if normalized in mapping:
        return mapping[normalized]

    return np.nan


def prepare_uploaded_frame(
    frame: pd.DataFrame,
    expected_features: Iterable[str],
) -> PreparedDataset:
    """Normalize columns, validate features, and parse an optional target."""
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows.")

    normalized = frame.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]

    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError(
            "Two or more columns become identical after name normalization. "
            "Rename duplicate columns and upload the file again."
        )

    expected = [normalize_column_name(feature) for feature in expected_features]
    missing = [feature for feature in expected if feature not in normalized.columns]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" and {len(missing) - 8} more"
        raise ValueError(f"Missing required feature columns: {preview}{suffix}.")

    target_candidates = [
        column
        for column in ("target", "diagnosis", "label", "class", "outcome")
        if column in normalized.columns
    ]
    if len(target_candidates) > 1:
        raise ValueError(
            "Multiple possible target columns were found: "
            + ", ".join(target_candidates)
            + ". Keep only one target column."
        )

    target_column = target_candidates[0] if target_candidates else None
    target: pd.Series | None = None
    warnings: list[str] = []

    if target_column is not None:
        mapped_target = normalized[target_column].map(_map_target_value)
        if mapped_target.isna().any():
            invalid_count = int(mapped_target.isna().sum())
            raise ValueError(
                f"The target column contains {invalid_count} unsupported or missing values. "
                "Use 0/1, B/M, or benign/malignant."
            )
        target = mapped_target.astype(int).rename("target")

    feature_frame = normalized.loc[:, expected].copy()
    for column in expected:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
        if feature_frame[column].isna().all():
            raise ValueError(
                f"Feature '{column}' has no usable numeric values after conversion."
            )

    missing_cells = int(feature_frame.isna().sum().sum())
    if missing_cells:
        warnings.append(
            f"The feature matrix contains {missing_cells} missing or nonnumeric values. "
            "The saved pipelines will impute them with training-set medians."
        )

    known_non_features = {
        "id",
        "sample_id",
        "record_id",
        "target_label",
        "prediction",
        "predicted_target",
        "predicted_label",
        "malignant_probability",
    }
    ignored = [
        column
        for column in normalized.columns
        if column not in expected
        and column != target_column
        and (column in known_non_features or column.startswith("unnamed"))
    ]
    unexpected = [
        column
        for column in normalized.columns
        if column not in expected and column != target_column and column not in ignored
    ]
    if unexpected:
        warnings.append(
            "Extra columns were ignored: " + ", ".join(unexpected[:10])
            + (" ..." if len(unexpected) > 10 else "")
        )
        ignored.extend(unexpected)

    return PreparedDataset(
        features=feature_frame,
        target=target,
        normalized_frame=normalized,
        target_column=target_column,
        ignored_columns=ignored,
        warnings=warnings,
    )
