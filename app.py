"""Interactive Streamlit application for Assignment 2.

Local run command:
    streamlit run app.py

Expected result:
    A browser opens with model selection, metrics, comparison, confusion matrix,
    classification report, predictions, and CSV download controls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from model.model_registry import MODEL_DISPLAY_ORDER, MODEL_REGISTRY
from utils.data_validation import PreparedDataset, prepare_uploaded_frame
from utils.evaluation import (
    METRIC_COLUMNS,
    classification_report_frame,
    confusion_values,
    evaluate_classifier,
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model" / "artifacts"
DEFAULT_TEST_FILE = ROOT / "test_data.csv"
CONFIG_FILE = ROOT / "student_config.json"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"

st.set_page_config(
    page_title="TumorScope ML",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
    .hero-card {
        border: 1px solid rgba(49, 91, 191, 0.18);
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        background: linear-gradient(135deg, rgba(52, 97, 212, 0.10), rgba(28, 167, 146, 0.08));
        margin-bottom: 1.1rem;
    }
    .hero-title {font-size: 2rem; font-weight: 750; margin-bottom: 0.2rem;}
    .hero-subtitle {font-size: 1rem; opacity: 0.83; margin-bottom: 0;}
    .section-note {
        border-left: 4px solid #315bbf;
        padding: 0.7rem 0.9rem;
        background: rgba(49, 91, 191, 0.06);
        border-radius: 0 10px 10px 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 130, 150, 0.18);
        border-radius: 12px;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.55);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_saved_model(model_name: str) -> Any:
    artifact_name = MODEL_REGISTRY[model_name]["artifact"]
    artifact_path = ARTIFACT_DIR / artifact_name
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {artifact_path}. Run python -m model.train_models."
        )
    return joblib.load(artifact_path)


def read_student_config() -> dict[str, str]:
    defaults = {
        "student_name": "REPLACE WITH YOUR NAME",
        "student_id": "REPLACE WITH YOUR BITS ID",
        "github_url": "REPLACE WITH YOUR GITHUB REPOSITORY URL",
        "streamlit_url": "REPLACE WITH YOUR LIVE STREAMLIT URL",
    }
    if not CONFIG_FILE.exists():
        return defaults
    try:
        loaded = load_json(CONFIG_FILE)
    except (json.JSONDecodeError, OSError):
        return defaults
    return {key: str(loaded.get(key, value)) for key, value in defaults.items()}


def format_metric(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def render_metric_cards(metrics: dict[str, float]) -> None:
    first_row = st.columns(3)
    second_row = st.columns(3)
    for column, metric_name in zip(first_row + second_row, METRIC_COLUMNS):
        column.metric(metric_name, format_metric(metrics[metric_name]))


def confusion_matrix_figure(target: pd.Series, predictions: np.ndarray) -> plt.Figure:
    matrix = confusion_values(target, predictions)
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Benign", "Malignant"],
    )
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    return figure


def prediction_frame(
    prepared: PreparedDataset,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    output = prepared.normalized_frame.copy()
    output["predicted_target"] = predictions.astype(int)
    output["predicted_label"] = np.where(
        predictions.astype(int) == 1, "Malignant", "Benign"
    )
    output["malignant_probability"] = scores
    return output


def evaluate_all_models(
    prepared: PreparedDataset,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    if prepared.target is None:
        raise ValueError("A target column is required to compare model metrics.")

    rows: list[dict[str, object]] = []
    details: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODEL_DISPLAY_ORDER:
        model = load_saved_model(model_name)
        metrics, predictions, scores = evaluate_classifier(
            model, prepared.features, prepared.target
        )
        rows.append({"Model": model_name, **metrics})
        details[model_name] = (predictions, scores)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        by=["MCC", "F1", "AUC", "Accuracy"], ascending=False
    ).reset_index(drop=True)
    return comparison, details


def model_observation(model_name: str, row: pd.Series, rank: int) -> str:
    metric_fragment = (
        f"Accuracy {row['Accuracy']:.4f}, recall {row['Recall']:.4f}, "
        f"F1 {row['F1']:.4f}, and MCC {row['MCC']:.4f}."
    )
    family_note = MODEL_REGISTRY[model_name]["summary"]
    if rank == 1:
        return f"Top overall ranking on this uploaded test set. {metric_fragment} {family_note}"
    return f"Ranked {rank} on the MCC-first rule. {metric_fragment} {family_note}"


def render_sidebar(metadata: dict[str, Any], student: dict[str, str]) -> None:
    st.sidebar.title("Assignment Navigator")
    st.sidebar.markdown("**Student**")
    st.sidebar.write(student["student_name"])
    st.sidebar.caption(student["student_id"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset snapshot**")
    st.sidebar.write(f"Instances: {metadata['dataset_instances']}")
    st.sidebar.write(f"Features: {metadata['dataset_features']}")
    st.sidebar.write(f"Classes: {len(metadata['target_mapping'])}")
    st.sidebar.write(f"Saved models: {len(metadata['models'])}")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Educational demonstration only. This application is not a medical diagnostic tool."
    )


def dataset_input(metadata: dict[str, Any]) -> tuple[PreparedDataset, str]:
    upload = st.file_uploader(
        "Upload test data as CSV",
        type=["csv"],
        help=(
            "Use the same 30 feature columns as test_data.csv. Add target, diagnosis, "
            "label, class, or outcome to calculate metrics."
        ),
    )

    if upload is None:
        raw_frame = load_csv(DEFAULT_TEST_FILE)
        source_label = "Bundled test_data.csv"
    else:
        try:
            raw_frame = pd.read_csv(upload)
        except Exception as exc:
            raise ValueError(f"The uploaded file could not be parsed as CSV: {exc}") from exc
        source_label = upload.name

    prepared = prepare_uploaded_frame(raw_frame, metadata["feature_names"])
    return prepared, source_label


def render_selected_model_tab(prepared: PreparedDataset) -> None:
    st.subheader("Selected model evaluation")
    selected_model = st.selectbox(
        "Choose a classifier",
        options=list(MODEL_DISPLAY_ORDER),
        index=0,
    )
    st.caption(
        f"{MODEL_REGISTRY[selected_model]['family']}: "
        f"{MODEL_REGISTRY[selected_model]['summary']}"
    )

    model = load_saved_model(selected_model)
    predictions = np.asarray(model.predict(prepared.features), dtype=int)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(prepared.features)[:, 1], dtype=float)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(prepared.features), dtype=float)
    else:
        scores = predictions.astype(float)

    if prepared.target is None:
        st.warning(
            "No target column was found. Predictions are available, but evaluation "
            "metrics, confusion matrix, and classification report require known labels."
        )
    else:
        metrics, predictions, scores = evaluate_classifier(
            model, prepared.features, prepared.target
        )
        render_metric_cards(metrics)

        left, right = st.columns([1, 1.2])
        with left:
            figure = confusion_matrix_figure(prepared.target, predictions)
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
        with right:
            st.markdown("#### Classification report")
            report = classification_report_frame(prepared.target, predictions)
            st.dataframe(
                report.style.format(
                    {
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1-score": "{:.4f}",
                        "support": "{:.0f}",
                    }
                ),
                use_container_width=True,
            )

    st.markdown("#### Row-level predictions")
    output = prediction_frame(prepared, predictions, scores)
    preview_columns = [
        column
        for column in (
            prepared.target_column,
            "predicted_target",
            "predicted_label",
            "malignant_probability",
        )
        if column is not None and column in output.columns
    ]
    st.dataframe(output.loc[:, preview_columns].head(25), use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )


def render_comparison_tab(prepared: PreparedDataset) -> None:
    st.subheader("All-model comparison")
    if prepared.target is None:
        st.info("Add a target column to compare model performance.")
        return

    comparison, _ = evaluate_all_models(prepared)
    formatted = comparison.copy()
    for metric in METRIC_COLUMNS:
        formatted[metric] = formatted[metric].map(
            lambda value: None if pd.isna(value) else round(float(value), 4)
        )

    st.dataframe(
        formatted.style.highlight_max(subset=list(METRIC_COLUMNS), axis=0),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Visual comparison")
    chart_frame = comparison.set_index("Model")[["Accuracy", "AUC", "F1", "MCC"]]
    st.bar_chart(chart_frame)

    winner = str(comparison.loc[0, "Model"])
    st.success(
        f"Overall winner for this test set: {winner}. "
        "Ranking uses MCC first, followed by F1, AUC, and Accuracy."
    )

    observation_rows = []
    for rank, row in comparison.iterrows():
        model_name = str(row["Model"])
        observation_rows.append(
            {
                "ML Model Name": model_name,
                "Observation about model performance": model_observation(
                    model_name, row, rank + 1
                ),
            }
        )
    st.dataframe(pd.DataFrame(observation_rows), use_container_width=True, hide_index=True)


def render_dataset_tab(metadata: dict[str, Any]) -> None:
    st.subheader("Dataset and experiment design")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Dataset")
        st.write(metadata["dataset_name"])
        st.write(
            f"The project uses {metadata['dataset_instances']} rows and "
            f"{metadata['dataset_features']} numeric features."
        )
        st.write(
            f"Training rows: {metadata['train_rows']} | Test rows: {metadata['test_rows']}"
        )
        st.link_button("Open the UCI dataset page", metadata["dataset_source"])
    with right:
        st.markdown("#### Target mapping")
        st.write("0 = Benign")
        st.write("1 = Malignant")
        st.write("Positive class for Precision, Recall, F1, and AUC: Malignant (1)")
        st.write(f"Random state: {metadata['random_state']}")

    st.markdown("#### Feature dictionary")
    dictionary = load_csv(ROOT / "data" / "feature_dictionary.csv")
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

    st.download_button(
        "Download bundled test_data.csv",
        data=DEFAULT_TEST_FILE.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )


def main() -> None:
    if not METADATA_FILE.exists():
        st.error(
            "Model metadata is missing. Run `python -m model.train_models` from the "
            "project root, then restart Streamlit."
        )
        st.stop()

    metadata = load_json(METADATA_FILE)
    student = read_student_config()
    render_sidebar(metadata, student)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">TumorScope ML Classification Studio</div>
            <div class="hero-subtitle">
                Compare six classification pipelines on the Breast Cancer Wisconsin
                Diagnostic test set, inspect evaluation metrics, and export predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        prepared, source_label = dataset_input(metadata)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    for warning in prepared.warnings:
        st.warning(warning)

    data_col, missing_col, class_col = st.columns(3)
    data_col.metric("Rows loaded", f"{prepared.features.shape[0]}")
    missing_col.metric("Missing feature cells", f"{int(prepared.features.isna().sum().sum())}")
    class_text = (
        "No labels"
        if prepared.target is None
        else f"{prepared.target.nunique()} classes"
    )
    class_col.metric("Target status", class_text)
    st.caption(f"Current data source: {source_label}")

    with st.expander("Preview normalized input data", expanded=False):
        st.dataframe(prepared.normalized_frame.head(20), use_container_width=True)

    selected_tab, comparison_tab, dataset_tab = st.tabs(
        ["Evaluate one model", "Compare all models", "Dataset guide"]
    )
    with selected_tab:
        render_selected_model_tab(prepared)
    with comparison_tab:
        render_comparison_tab(prepared)
    with dataset_tab:
        render_dataset_tab(metadata)


if __name__ == "__main__":
    main()
