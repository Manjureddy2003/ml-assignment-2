# Personalize Before Submission

Do not submit the ZIP unchanged. Complete each item below honestly.

## Required identity and links

1. Edit `student_config.json` with your name, BITS ID, GitHub repository URL, and deployed Streamlit URL.
2. Replace the same placeholders in `README.md` and `report/submission_report.docx`.
3. Insert one genuine screenshot showing the assignment running on BITS Virtual Lab.
4. Export the updated DOCX as a single PDF and verify that all links are clickable.

## Make the work your own

1. Read and run `model/train_models.py`; be prepared to explain preprocessing, class mapping, split strategy, and each model.
2. Rewrite the performance observations in your own words after checking the actual output.
3. Change the Streamlit title, section wording, or layout so it reflects your presentation style.
4. Add one meaningful enhancement, such as a ROC curve, threshold control, feature-importance view, or error-analysis table.
5. Keep a genuine Git commit history while you test and improve the project. Do not manufacture or backdate commits.

## Final technical checks

```bash
python -m pip install -r requirements-dev.txt
python -m model.train_models
pytest -q
streamlit run app.py
```

Expected results:

- Model training prints a six-row comparison table.
- Tests report `6 passed`.
- Streamlit opens without errors.
- Uploading `test_data.csv` displays all six required metrics, a confusion matrix, and a classification report.

## Final submission checks

- GitHub repository is public or accessible to the evaluator.
- `requirements.txt` is in the repository root.
- `test_data.csv` is in the repository root.
- All model artifacts are present under `model/artifacts/`.
- Live Streamlit link opens in a private/incognito browser window.
- The submitted PDF includes the GitHub link, live app link, README content, model table, observations, and BITS Virtual Lab screenshot.
