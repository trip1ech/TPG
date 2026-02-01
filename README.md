# TPG — Bus Segment Travel Time Modeling & Dashboard

This repository contains notebooks and a small Streamlit app for analyzing and modeling TPG (Geneva) bus segment travel time, including a 3-stage ablation study:

- Stage 1 (Static): baseline + calendar/time features
- Stage 2 (Network): adds network/segment context
- Stage 3 (Full): adds signal-related/context features (when available)

It also includes a Streamlit dashboard to explore model predictions spatially and temporally.

## Repository layout

analysis/            Jupyter notebooks used for the main analysis
ben_reproduction/    Notebook reproducing Ben’s original work + related artifacts
app/                 Streamlit dashboard code
app_data/            Inputs/outputs used by the app (predictions, stops, weather)
notebook-archive/    Experimental / scratch notebooks
scripts/             Utility Python scripts

data/                Raw/intermediate datasets (ignored by git)
models/              Trained LightGBM models (ignored by git)
figure/              Figures/exports (optional)
related-paper/       Paper notes and related references

## Quickstart

1) Create an environment (conda example)

    conda create -n tpg-ml python=3.11
    conda activate tpg-ml

2) Install app dependencies

    pip install -r app/requirements.txt

For running notebooks you will likely also need (depending on your workflow):
- jupyter / ipykernel
- lightgbm
- scikit-learn
- matplotlib

## Running the dashboard

    streamlit run app/app.py

### App inputs

The dashboard expects these files under app_data/:
- pred_stage1_test.parquet
- pred_stage2_test.parquet
- pred_stage3_test.parquet
- stops_df.csv
- weather-info.csv

If these files are missing, generate them from the analysis notebooks.

## Notebooks

- analysis/analyisis-full-data.ipynb: main end-to-end notebook (data prep → features → training/evaluation → saved predictions/models)
- analysis/signal-processing.ipynb
- analysis/tpg-statistics.ipynb
- ben_reproduction/ben_reproduction.ipynb: reproduction of Ben’s work
- notebook-archive/: scratch/experimental notebooks

## Models & saved artifacts

The training pipeline saves LightGBM boosters and per-row prediction tables:

- Models: models/bst_stage1.txt, models/bst_stage2.txt, models/bst_stage3.txt
- App predictions: app_data/pred_stage*_test.parquet

Note: the current .gitignore intentionally ignores many large files and folders (including data/ and models/, and many tabular formats like *.csv).

If you want to version models or app inputs:
- consider Git LFS for large artifacts, or
- adjust .gitignore to allow specific tracked files.

## Scripts

- scripts/prepare_map_data.py: helper script(s) used during data preparation
