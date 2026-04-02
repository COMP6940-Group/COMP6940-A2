# Part 1: GPU-Accelerated Weibull Fitting

This folder contains the notebook for **Part 1** of COMP6940 Assignment 2, where a 2-parameter Weibull distribution is fitted to engine run-to-failure data using maximum likelihood estimation (MLE), with emphasis on GPU-oriented computation.

> [!NOTE]
> The JAX implementation in this project was run on an Apple M1 Pro machine. JAX does not provide native Apple GPU (Metal) support in its standard platform matrix, so execution here should be treated as CPU-based. Google Colab's free T4 GPU tier was tested but produced worse wall-clock times than the CPU implementation, which is likely due to the overhead of transferring data between the CPU and GPU and the small size of the dataset.
## Objective

- Extract engine time-to-failure values from NASA C-MAPSS FD001 training data.
- Fit Weibull parameters (`k` shape, `lambda` scale) using MLE.
- Implement gradient-based optimization with JAX.
- Compare/benchmark computational workflow (GPU-oriented vs CPU baseline where applicable).
- Interpret reliability outcomes for maintenance decision-making.

## Data

- **Source:** NASA C-MAPSS FD001 (`train_FD001.txt`)
- **Expected raw location:** `../data/raw/cmapss/train_FD001.txt`
- **Core variable used for fitting:** per-engine failure cycle (max cycle per unit)

## Notebook

- `part1_gpu_weibull.ipynb`

The notebook is expected to cover:

1. **Ingestion and preprocessing**
   - Load FD001 data and assign column names.
   - Aggregate by engine unit to obtain time-to-failure samples.
   - Report summary statistics (mean/median/min/max).
2. **Weibull MLE implementation**
   - Define Weibull log-likelihood.
   - Optimize negative log-likelihood with JAX-based gradients.
   - Report fitted parameters (`k`, `lambda`) and training time.
3. **Visualization and interpretation**
   - Histogram of failure times with fitted Weibull PDF overlay.
   - Survival curve with key reliability markers (e.g., 90% survival and median life).
   - Maintenance recommendation and fit-quality discussion.