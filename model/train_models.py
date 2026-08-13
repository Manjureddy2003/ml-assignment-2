"""Train all assignment classifiers and create reproducible project artifacts.

Expected console output ends with a six-row metric table and:
    Training complete. Artifacts saved under model/artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.model_registry import MODEL_DISPLAY_ORDER, MODEL_REGISTRY
from utils.data_validation import normalize_column_name
from utils.evaluation import evaluate_classifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "model" / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATASET_SOURCE = (
    "https://archive.ics.uci.edu/dataset/17/"
    "breast+cancer+wisconsin+diagnostic"
)


def build_models() -> dict[str, Pipeline]:
    """Construct the six classifiers used in the project."""

    def scaled_steps() -> list[tuple[str, object]]:
        return [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

    return {
        "Logistic Regression": Pipeline(
            scaled_steps()
            + [
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            scaled_steps()
            + [
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="minkowski",
                    ),
                )
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            scaled_steps()
            + [("classifier", GaussianNB(var_smoothing=1e-9))]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            scaled_steps()
            + [
                (
                    "classifier",
                    SVC(
                        C=2.0,
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def _measurement_description(feature_name: str) -> tuple[str, str]:
    if feature_name.startswith("mean_"):
        return "Mean", "Mean value computed across the segmented cell nuclei."
    if feature_name.startswith("worst_"):
        return "Worst", "Largest or most extreme value among the measured nuclei."
    if feature_name.endswith("_error"):
        return "Standard Error", "Standard error of the corresponding measurement."
    return "Other", "Numeric diagnostic feature supplied with the dataset."


def _write_feature_dictionary(feature_names: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for feature in feature_names:
        group, description = _measurement_description(feature)
        rows.append(
            {
                "feature_name": feature,
                "feature_group": group,
                "description": description,
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "feature_dictionary.csv", index=False)


def _safe_metric_value(value: float) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 8)


def train_and_save() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    features.columns = [normalize_column_name(name) for name in features.columns]

    # The scikit-learn copy uses 0=malignant and 1=benign. This project remaps
    # the positive class to 1=malignant so precision, recall, F1, and AUC focus
    # on malignant-case detection.
    target = (1 - dataset.target.astype(int)).rename("target")

    full_frame = features.copy()
    full_frame["target"] = target

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    train_frame = train_features.copy()
    train_frame["target"] = train_target
    test_frame = test_features.copy()
    test_frame["target"] = test_target

    full_frame.to_csv(DATA_DIR / "breast_cancer_wisconsin_full.csv", index=False)
    train_frame.to_csv(DATA_DIR / "train_data.csv", index=False)
    test_frame.to_csv(ROOT / "test_data.csv", index=False)
    test_frame.to_csv(DATA_DIR / "test_data.csv", index=False)
    _write_feature_dictionary(list(features.columns))

    models = build_models()
    metric_rows: list[dict[str, object]] = []
    reports: dict[str, object] = {}

    for model_name in MODEL_DISPLAY_ORDER:
        model = models[model_name]
        model.fit(train_features, train_target)
        metrics, predictions, scores = evaluate_classifier(
            model, test_features, test_target
        )

        artifact_path = ARTIFACT_DIR / MODEL_REGISTRY[model_name]["artifact"]
        joblib.dump(model, artifact_path, compress=3)

        metric_rows.append({"Model": model_name, **metrics})
        reports[model_name] = {
            "predictions": [int(value) for value in predictions.tolist()],
            "positive_scores": [round(float(value), 8) for value in scores.tolist()],
        }

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(ARTIFACT_DIR / "metrics.csv", index=False)

    ranked = metrics_frame.sort_values(
        by=["MCC", "F1", "AUC", "Accuracy"], ascending=False
    ).reset_index(drop=True)
    winner = str(ranked.loc[0, "Model"])

    metadata = {
        "project_title": "TumorScope ML Classification Studio",
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": DATASET_SOURCE,
        "dataset_instances": int(full_frame.shape[0]),
        "dataset_features": int(features.shape[1]),
        "target_column": "target",
        "target_mapping": {"0": "Benign", "1": "Malignant"},
        "positive_class": "Malignant (1)",
        "feature_names": list(features.columns),
        "train_rows": int(train_frame.shape[0]),
        "test_rows": int(test_frame.shape[0]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "models": list(MODEL_DISPLAY_ORDER),
        "overall_winner_rule": "Highest MCC, then F1, AUC, and Accuracy",
        "overall_winner": winner,
        "sklearn_version": sklearn.__version__,
        "metrics": {
            row["Model"]: {
                key: _safe_metric_value(float(row[key]))
                for key in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
            }
            for row in metric_rows
        },
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "prediction_details.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    source_text = (
        "Dataset: Breast Cancer Wisconsin (Diagnostic)\n"
        f"Source: {DATASET_SOURCE}\n"
        "Rows: 569\n"
        "Features: 30 real-valued predictors\n"
        "Project target mapping: 0=Benign, 1=Malignant\n"
    )
    (DATA_DIR / "dataset_source.txt").write_text(source_text, encoding="utf-8")

    return metrics_frame


def main() -> None:
    metrics = train_and_save()
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTraining complete. Artifacts saved under model/artifacts.")


if __name__ == "__main__":
    main()
