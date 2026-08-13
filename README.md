# TumorScope ML Classification Studio

An end-to-end machine learning classification project for Machine Learning Assignment 2. The project trains six classifiers on one public dataset, evaluates them using all required metrics, and exposes the saved models through an interactive Streamlit application.

> Important: Replace the student details, GitHub URL, Streamlit URL, and BITS Virtual Lab screenshot before submission. Review the code and write the final observations in your own words.

## a. Problem statement

Build and compare multiple classification models on the same public dataset, calculate Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC), and deploy an interactive Streamlit application that accepts test CSV data and displays model results.

The assignment brief explicitly names five mandatory models but also refers to six models in two places. This project includes all five named models and adds a Support Vector Machine as a sixth model so that both interpretations are covered.

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Public source:** UCI Machine Learning Repository  
**Source URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
**Task:** Binary classification  
**Instances:** 569  
**Predictor features:** 30 real-valued features  
**Missing values in original data:** None  
**Training rows:** 455  
**Test rows:** 114  
**Split:** Stratified 80:20 split with `random_state=42`

The features describe characteristics computed from digitized images of breast-mass cell nuclei. The project uses the following target mapping:

| Target value | Class label |
|---:|---|
| 0 | Benign |
| 1 | Malignant |

The positive class for Precision, Recall, F1, and AUC is **Malignant (1)**.

## c. Submission links

- **GitHub Repository Link:** https://github.com/Manjureddy2003/ml-assignment-2
- **Live Streamlit App Link:** https://manjureddy2003-ml-assignment-2-app-hghjpb.streamlit.app/
- **BITS Virtual Lab Screenshot:** Insert the screenshot in `report/submission_report.docx`, then regenerate the PDF.

## d. Models used and comparison table

The following six models are implemented on the same train-test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier
6. Support Vector Machine with an RBF kernel

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | 0.9954 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9035 | 0.8980 | 0.8974 | 0.8333 | 0.8642 | 0.7908 |
| K-Nearest Neighbors | 0.9561 | 0.9835 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest | 0.9737 | 0.9974 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Support Vector Machine | **0.9825** | 0.9950 | **1.0000** | **0.9524** | **0.9756** | **0.9626** |

### Model-performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline. It achieved 0.9737 accuracy, 0.9524 malignant-class recall, and 0.9433 MCC after standardization. |
| Decision Tree | Lowest overall result on this split. Its lower recall and MCC suggest that a single depth-controlled tree did not generalize as well as the other models. |
| K-Nearest Neighbors | Good performance after scaling, with high precision and a 0.9058 MCC, but it missed more malignant cases than Logistic Regression and SVM. |
| Gaussian Naive Bayes | Very high AUC but lower thresholded recall and F1. The independence assumption limits its final class decisions even though its ranking scores are strong. |
| Random Forest | Best AUC at 0.9974 and perfect malignant-class precision on this test split. It produced no false-positive malignant predictions but missed three malignant rows. |
| Support Vector Machine | Best overall balance on the MCC-first ranking. It achieved the highest accuracy, F1, and MCC, with perfect precision and high recall. |
| Overall winner | **Support Vector Machine**, selected by highest MCC, followed by F1, AUC, and Accuracy as tie-breakers. Random Forest remains the AUC winner. |

These values are reproducible from the included `test_data.csv` and saved model artifacts. They may differ if the split, model settings, package versions, or dataset are changed.

## Streamlit application features

The application implements the required interactive behavior:

- CSV upload option; the bundled `test_data.csv` is used when no file is uploaded.
- Model-selection dropdown covering all six classifiers.
- Display of Accuracy, AUC, Precision, Recall, F1, and MCC.
- Confusion matrix and classification report.
- All-model comparison table and performance chart.
- Row-level predictions with malignant-class probabilities.
- Download button for the prediction CSV.
- Dataset guide and feature dictionary.
- Validation for missing columns, text target labels, extra columns, and missing numeric cells.

## Repository structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- requirements-dev.txt
|-- README.md
|-- test_data.csv
|-- student_config.json
|-- .streamlit/
|   `-- config.toml
|-- data/
|   |-- breast_cancer_wisconsin_full.csv
|   |-- train_data.csv
|   |-- test_data.csv
|   |-- feature_dictionary.csv
|   `-- dataset_source.txt
|-- model/
|   |-- train_models.py
|   |-- model_registry.py
|   `-- artifacts/
|       |-- logistic_regression.joblib
|       |-- decision_tree.joblib
|       |-- knn.joblib
|       |-- gaussian_naive_bayes.joblib
|       |-- random_forest.joblib
|       |-- support_vector_machine.joblib
|       |-- metrics.csv
|       `-- metadata.json
|-- utils/
|   |-- data_validation.py
|   `-- evaluation.py
|-- tests/
|   |-- test_pipeline.py
|   `-- test_data_validation.py
|-- tools/
|   `-- build_submission_report.py
|-- report/
|   |-- submission_report.docx
|   `-- submission_report.pdf
|-- DEPLOYMENT_GUIDE.md
`-- PERSONALIZE_BEFORE_SUBMISSION.md
```

## Local setup

Use the same Python version locally and on Streamlit Community Cloud. Python 3.13 and scikit-learn 1.8.0 were used to create the included model artifacts.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m model.train_models
streamlit run app.py
```

Expected training output:

```text
                 Model  Accuracy    AUC  Precision  Recall     F1    MCC
   Logistic Regression    0.9737 0.9954     0.9756  0.9524 0.9639 0.9433
         Decision Tree    0.9035 0.8980     0.8974  0.8333 0.8642 0.7908
   K-Nearest Neighbors    0.9561 0.9835     0.9744  0.9048 0.9383 0.9058
  Gaussian Naive Bayes    0.9211 0.9891     0.9231  0.8571 0.8889 0.8292
         Random Forest    0.9737 0.9974     1.0000  0.9286 0.9630 0.9442
Support Vector Machine    0.9825 0.9950     1.0000  0.9524 0.9756 0.9626

Training complete. Artifacts saved under model/artifacts.
```

Expected Streamlit result: a browser page titled **TumorScope ML Classification Studio** with the upload control, model dropdown, six metric cards, confusion matrix, classification report, comparison table, and prediction download button.

## Run automated tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Expected output:

```text
6 passed
```

## Deployment summary

1. Create a new GitHub repository.
2. Upload all project files while preserving the folder structure.
3. Make several meaningful commits as you review, test, personalize, and improve the project.
4. Open Streamlit Community Cloud and create an app from the repository.
5. Select `app.py` as the entrypoint and use the same Python version as local development.
6. Deploy, test `test_data.csv`, and copy the live URL.
7. Replace the links in `student_config.json` and the report.
8. Run the project on BITS Virtual Lab and capture one genuine execution screenshot.

Detailed instructions are in `DEPLOYMENT_GUIDE.md`.

## Rebuild models

Run this whenever you change the data split or model settings:

```bash
python -m model.train_models
```

This regenerates the train/test CSV files, all six `.joblib` artifacts, `metrics.csv`, and `metadata.json`.

## Notes and limitations

- This is an educational machine learning demonstration and must not be used for medical diagnosis.
- The test set contains 114 rows, so a small number of classification changes can noticeably affect the reported metrics.
- AUC uses malignant-class scores; other binary metrics also treat Malignant (1) as the positive class.
- The included report still needs your name, BITS ID, final links, and BITS Virtual Lab screenshot.
- Academic integrity checks may compare code structure, variable names, outputs, and UI. Personalize the application, understand every section, and submit only work you can explain.
