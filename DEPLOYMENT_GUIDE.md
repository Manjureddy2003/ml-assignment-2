# Deployment Guide

## 1. Verify locally

```bash
python -m pip install -r requirements.txt
python -m model.train_models
streamlit run app.py
```

Expected result: the application opens locally and the bundled `test_data.csv` produces metrics for all six models.

## 2. Create the GitHub repository

Create a new repository and add the complete project folder. Preserve the relative paths because `app.py` loads data and model artifacts from them.

Example commands:

```bash
git init
git add README.md requirements.txt app.py student_config.json
git commit -m "Create assignment project structure"

git add data model utils test_data.csv
git commit -m "Add dataset processing and six trained classifiers"

git add tests
git commit -m "Add reproducibility and validation tests"

git add .streamlit DEPLOYMENT_GUIDE.md report
git commit -m "Add Streamlit configuration and submission report"
```

Use commits that match work you actually performed. Review and modify the project between commits.

## 3. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app.
3. Select your repository and branch.
4. Set the entrypoint file to `app.py`.
5. In advanced settings, select the same Python version used locally. The included artifacts were created with Python 3.13 and scikit-learn 1.8.0.
6. Deploy and watch the build log.

The root `requirements.txt` lists every nonstandard package imported by the app.

## 4. Validate the live app

- Open the live URL in an incognito/private window.
- Confirm the bundled test data loads automatically.
- Select each model.
- Upload `test_data.csv` manually.
- Confirm the six metrics, confusion matrix, classification report, comparison table, and download button work.
- Download predictions and verify the CSV opens.

## 5. Update submission links

Edit `student_config.json`:

```json
{
  "student_name": "Your Name",
  "student_id": "Your BITS ID",
  "github_url": "https://github.com/your-user/your-repository",
  "streamlit_url": "https://your-app-name.streamlit.app"
}
```

Expected result: your identity appears in the app sidebar after the updated file is pushed.

## 6. Capture the required BITS Virtual Lab screenshot

Run the application inside BITS Virtual Lab and capture one screenshot that visibly shows the lab environment and successful assignment execution. Do not use a locally fabricated screenshot.

Save it as:

```text
report/bits_virtual_lab_screenshot.png
```

Then rebuild the report:

```bash
python tools/build_submission_report.py
```

If LibreOffice is available, export `report/submission_report.docx` to PDF. Open the PDF and verify the screenshot, tables, and links before submitting.

## Troubleshooting

### Build fails while installing dependencies

Confirm `requirements.txt` is in the repository root and the selected Python version is compatible with the pinned scikit-learn version.

### Model artifact cannot be loaded

Run `python -m model.train_models`, commit the regenerated files under `model/artifacts/`, and redeploy.

### Uploaded CSV reports missing features

Use the exact 30 feature columns in `test_data.csv`. Column names are normalized to lowercase snake_case, but all required predictors must be present.

### Metrics show N/A for AUC

The uploaded target contains only one class. AUC requires both benign and malignant labels in the evaluation file.
