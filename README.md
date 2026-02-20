# Project: EDA & Modeling — Lectures 2, 3, 4 and Integrated Pipeline

## Project Summary
This repository collects coursework and code for Lectures 2, 3 and 4 and an integrated pipeline that ties the material together. It includes exploratory data analysis (EDA), feature engineering, transformations, visualization, and machine learning experiments applied primarily to a movies dataset (downloaded with `kagglehub`). The repo is organized so you can reproduce individual lecture deliverables and run the full pipeline that consolidates them.

## Contents (high level)
- `Lect2/` — materials for Lecture 2 (models, evaluation examples). See `Lect2/README.md` if present.
- `Lect3/` — intermediate analysis and experiments (synthetic fintech dataset examples: `fintech_top_sintetico_2025.csv`, `lab_fintech_sintetico_2025.py`, `mercado_miguel_fintech_analysis.py`).
- `Lect4/` — EDA and pipeline scripts for the movies dataset. Key files:
  - `Lect4/lect4.py` — EDA and initial transformations (histograms, scatter plots, basic encodings).
  - `Lect4/movies_complete_pipeline.py` — integrated pipeline: preprocessing, feature engineering, model training, CV, reporting, and model serialization.
- `Informe1/` — report and documentation files for the project. See `Informe1/README.md` (a short guide) and `Informe1/PIPELINE_COMPLETO_README.md` or `Informe1/README.md` for the pipeline-focused documentation.
- `outputs/` — generated artifacts (plots, CSV reports, serialized model files). Example items include `pipeline_histogramas.png`, `pipeline_resultados_modelos.csv`, `best_model_movies.joblib`, and `analisis_conclusiones.txt`.

## Key artifacts produced by the work
- EDA images and interactive HTMLs (if generated): histograms, scatter matrices, PCA plots, TSNE, correlation heatmaps.
- Text summaries: `analisis_conclusiones.txt`, `pipeline_reporte_completo.txt`.
- Model outputs: `pipeline_resultados_modelos.csv`, `best_model_movies.joblib`.
- Documentation: `Informe1/README.md` and the project-level `README.md` you are reading.

## How this maps to course lectures
- Lecture 2: model training basics, evaluation metrics and simple baseline models. See `Lect2/`.
- Lecture 3: intermediate feature/engineering, synthetic dataset experiments, and model diagnostics. See `Lect3/`.
- Lecture 4: advanced EDA and preparing datasets for modeling, including transformations, encodings, and visualizations. See `Lect4/`.
- Integrated Pipeline: `Lect4/movies_complete_pipeline.py` demonstrates an end-to-end flow combining EDA, preprocessing, feature engineering, dimensionality reduction for visualization, model training/evaluation, and artifact saving.

## Quick reproduction steps
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies (if `requirements.txt` exists) or install core libraries:

```bash
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly joblib kagglehub
```

3. Ensure `outputs/` exists:

```bash
mkdir -p outputs
```

4. Run individual lecture scripts or the full pipeline:

```bash
# EDA for Lecture 4
python Lect4/lect4.py

# Full integrated pipeline
python Lect4/movies_complete_pipeline.py
```

You can also use the convenience script `run_pipeline.sh` if present.

## Notes on dataset location
- The movies dataset was downloaded using `kagglehub` and cached under the user cache path (e.g., `~/.cache/kagglehub/...`). `Lect4/` scripts attempt to read `movies.csv` from the local cache or the working directory. If the pipeline cannot find the data, download the dataset manually (or run `Lect4/lect4.py` to fetch it) and place `movies.csv` in the project root or update the script path.

## Results & caveats (summary)
- The pipeline run saved in this workspace reports a dataset reduced from 9,999 rows to 6,526 rows after IQR-based outlier removal, and trained multiple regression models.
- One or more runs reported very high model performance (R² ≈ 1.0). This is suspicious and likely caused by data leakage or feature-target entanglement. Before trusting a model in production, perform the following checks:
  - Ensure no features are derived using the target or future information.
  - Confirm the train/test split is strictly held out and that scaling/encoding is fit only on training data.
  - Run permutation importance or SHAP to audit feature influence.

## Recommended next steps (for the repo)
- Add or freeze a `requirements.txt` with exact versions used by your virtual environment.
- Add a short `README.md` inside each lecture folder describing the learning goals and which script to run first.
- Run a leakage/feature-audit pass on `Lect4/movies_complete_pipeline.py` (I can do this for you).
- Add unit tests or small integration tests to validate data schema assumptions (column names, dtypes, non-nullable columns).

## Where to find help in this repo
- `Lect4/movies_complete_pipeline.py` — main end-to-end script to inspect for the overall workflow.
- `Informe1/README.md` — project-specific report and guidance (already created).
- `outputs/` — inspect artifacts produced by running the scripts.

---

If you want, I will now:
- (A) Replace this file as the top-level `README.md` (already created),
- (B) Generate a `requirements.txt` from your venv, or
- (C) Run a leakage/feature-audit pass on `Lect4/movies_complete_pipeline.py` and produce a short remediation patch.

Tell me which option to do next (A / B / C) and I will proceed.

*Generated on: February 17, 2026*
