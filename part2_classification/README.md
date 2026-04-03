# Part 2: Classification, Impact Simulation, and Interpretability

This folder contains **Part 2** of COMP6940 Assignment 2. It builds a credit default prediction model, simulates business impact of deployment on held-out data, and interprets model decisions using SHAP, counterfactual explanations, and a fairness analysis.

## How to Run

From the project root:

```bash
uv sync
uv run jupyter notebook
```

> [!IMPORTANT]
> If you are on Apple Silicon and get the error `You are running 32-bit Python on a 64-bit OS`, make sure Homebrew has `libomp` installed:
>
> ```bash
> brew install libomp
> ```

Then open and run (in order):

1. `part2_classification/01_ingest_clean.ipynb`
2. `part2_classification/02_model_interpret.ipynb`

## Objective

- Clean and prepare the Home Credit default dataset.
- Train and evaluate multiple classification models (including at least one gradient-boosted tree model).
- Use the best model to estimate expected profit/loss at multiple decision thresholds and across a threshold sweep.
- Interpret the best model:
  - SHAP explanations (summary and individual drivers)
  - Counterfactual explanations (DiCE or equivalent)
  - Fairness analysis with at least one mitigation strategy and preliminary evidence

## Data

- **Source dataset:** Home Credit Default Risk (Kaggle)
- **Primary file:** `application_train.csv`
- **Expected raw path:** `../data/raw/home_credit/application_train.csv`
- **Expected curated output:**
  - `../data/curated/home_credit_curated.parquet`

## Notebooks

- `01_ingest_clean.ipynb`
  - Load `application_train.csv`
  - Report row counts, columns, dtypes, and top missingness
  - Apply cleaning rules:
    - drop columns with >40% missing values
    - impute numeric missing values with the median
    - impute categorical missing values with the mode
    - remove income outliers (beyond 99.5th percentile)
    - remove rows where `DAYS_EMPLOYED == 365243`
  - Feature engineering (engineered ratio features and derived fields)
  - Encode categorical variables (one-hot or ordinal with justification)
  - Save the curated dataset to `../data/curated/home_credit_curated.parquet`

- `02_model_interpret.ipynb`
  - Train/test split (80/20), stratified by `TARGET`
  - Handle class imbalance (class weights and/or random undersampling)
  - Train at least two models; evaluate and compare on the test set using:
    - ROC-AUC
    - MCC
    - precision
    - recall
    - F1-score
  - Impact simulation (using only the held-out test set):
    - expected profit/loss at thresholds `0.3`, `0.5`, `0.7`
    - profit curve for thresholds from `0.0` to `1.0` in `0.01` steps
    - compare against baselines (approve everyone; random classifier with same approval rate)
  - Interpretability & responsible use:
    - SHAP summary plot (top 15 features) and two waterfall plots
    - counterfactuals for three test-set defaulters in a side-by-side table
    - fairness analysis on at least one sensitive/protected attribute and at least one mitigation with preliminary experiments