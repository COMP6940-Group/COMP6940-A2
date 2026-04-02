# COMP6940 Assignment 2

This repository contains work for **COMP6940: Big Data and Data Visualization (UWI)**, Group Assignment 2.  
The assignment has two required parts:

- **Part 1:** GPU-accelerated Weibull fitting for turbofan engine time-to-failure data (C-MAPSS FD001).
- **Part 2:** Credit default classification with impact simulation and interpretability (SHAP and counterfactuals).

The project is organized to support reproducible notebook-based analysis with separate folders for raw and curated data.

## Assignment Scope (Brief)

### Part 1 - GPU Weibull MLE
- Load C-MAPSS run-to-failure data and extract per-engine failure times.
- Fit a 2-parameter Weibull distribution using MLE.
- Implement and benchmark a GPU-oriented approach (JAX), with comparison to CPU workflow.
- Analyze fit quality and reliability implications (survival behavior, maintenance recommendation).

### Part 2 - Classification and Interpretability
- Clean and engineer features from Home Credit data.
- Train and compare at least two models (including a gradient-boosted tree).
- Simulate business impact across decision thresholds (profit/loss analysis).
- Interpret model behavior using SHAP, counterfactual explanations, and fairness checks.

## Repository Structure

```text
COMP6940 A2/
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── data/
│   ├── raw/
│   │   └── cmapss/
│   │       └── train_FD001.txt
│   └── curated/
│       ├── home_credit_curated.parquet
│       └── model_comparison.csv
├── part1_gpu_weibull/
│   └── part1_gpu_weibull.ipynb
└── part2_classification/
    ├── 01_ingest_clean.ipynb
    └── 02_model_interpret.ipynb
```

## Data Sources

- **NASA C-MAPSS (FD001):** turbofan engine degradation / failure cycles.
- **Home Credit Default Risk:** tabular application data for default prediction.

Raw datasets should be stored under `data/raw/`, and processed outputs under `data/curated/`.

## Environment Setup

Recommended: use `uv` for fast, reproducible environment setup.

```bash
uv sync
```

Alternative: use `pip` with a virtual environment.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Order

1. Execute `part1_gpu_weibull/part1_gpu_weibull.ipynb`.
2. Execute `part2_classification/01_ingest_clean.ipynb`.
3. Execute `part2_classification/02_model_interpret.ipynb`.

Use notebook kernels tied to the same environment to ensure reproducibility.

## Notes

- This top-level README gives the project-wide overview.
- Part-specific READMEs for `part1_gpu_weibull/` and `part2_classification/` can be added next.
