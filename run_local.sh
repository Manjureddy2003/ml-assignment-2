#!/usr/bin/env sh
set -eu
python -m pip install -r requirements.txt
python -m model.train_models
streamlit run app.py
