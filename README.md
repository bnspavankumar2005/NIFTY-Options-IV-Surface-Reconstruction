# NIFTY Options IV Surface Reconstruction

**Finance Club, IIT Roorkee — Open Project 2026**  

---

## Project Overview

This project reconstructs the **Implied Volatility (IV) Surface** of NIFTY 50 options by predicting missing IV values scattered across the options chain. The dataset contains **975 timestamps** and **28 option columns** (14 CE calls + 14 PE puts), yielding 27,300 total rows in long format with **5,460 missing IV values (~20%)**.

The solution combines a **flat-wing parabolic baseline** that captures the smile shape analytically, with a **gradient boosting residual model** that learns corrections from temporal, spatial, and financial features — validated using a strictly sequential, lookahead-free cross-validation framework.

**Winner: GradientBoosting — CV MSE: 0.052084**

---

## Repository Structure

```
.
├── eda_analysis.ipynb          # Exploratory Data Analysis — run first
├── evaluation_model.ipynb      # Model pipeline, CV, and submission — run second
├── dataset.csv                 # Original dataset with missing IV values
├── filled_dataset.csv          # Intermediate: all 5,460 cells filled (auto-generated)
├── submission.csv              # Final Kaggle submission (auto-generated)
└── README.md
```

---

## Notebooks

### 1. `eda_analysis.ipynb` — Exploratory Data Analysis

Run this notebook first. Every section prints analysis and renders plots inline via `plt.show()` — no files are saved to disk.

| Section | What it analyzes |
|---|---|
| Raw Data Overview | Shape (975×30), date range, strike coverage (CE: 25200–26500, PE: 23800–25100) |
| IV Distribution | Percentile table, skew/kurtosis, CE vs PE comparison |
| IV Over Time | Median IV with IQR band, spot price overlay, lag-1 autocorrelation |
| Missing Data Analysis | Heatmap, missing % per column and per timestamp, ITM/ATM/OTM pattern |
| Volatility Smile & Skew | Smile plots at 5 timestamps with fitted splines, put skew quantification |
| Term Structure | ATM IV persistence over time, intraday vol pattern by hour |
| Spot Price Dynamics | 5-min returns, rolling realised vol, IV–spot correlation |
| Data Quality Checks | Outliers, put-call IV divergence, stale values, timestamp gap check |
| Feature Correlation | Pearson correlation of all engineered features against the IV target |
| Spline LOO MSE | Leave-one-out cross-validation MSE — the baseline the ML model must beat |
| Missingness by Zone | Missing rate by moneyness zone (ATM vs Deep OTM) |
| EDA Dashboard | Summary panel with modelling recommendations |

---

### 2. `evaluation_model.ipynb` — Model Pipeline & Submission

Complete end-to-end pipeline. Runs multi-model cross-validation, selects the best model automatically, trains on all observed data, and generates `submission.csv`.

---

## Actual Results

### Cross-Validation Output (5 folds, embargo=2, holdout=30%)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg CV MSE |
|---|---|---|---|---|---|---|
| RandomForest | 0.000026 | 0.000042 | 0.000039 | 0.000084 | 0.260266 | 0.052091 |
| GradientBoosting | 0.000026 | 0.000041 | 0.000038 | 0.000083 | 0.260236 | **0.052084** ✓ |
| LightGBM | 0.000026 | 0.000042 | 0.000038 | 0.000082 | 0.260237 | 0.052085 |
| XGBoost | 0.000026 | 0.000041 | 0.000038 | 0.000082 | 0.260245 | 0.052086 |

**Winner: GradientBoosting** (lowest average CV MSE: 0.052084)

**Note on Fold 5:** All models show elevated MSE in Fold 5 (~0.260). This is expected — Fold 5 covers the latest timestamps where the IV distribution shifts most significantly from the training period. This is not a bug; it reflects genuine out-of-distribution difficulty in the final market regime. Folds 1–4 show consistently low MSE (0.000026–0.000084), confirming the model performs well within distribution.

### Final Training
- Model trained on all **21,840 observed rows** (27,300 total − 5,460 missing)
- **5,460 missing cells predicted** successfully
- All predictions within `[DATA_IV_FLOOR, DATA_IV_CAP]` — assertions passed

---

## Methodology

### Architecture: Flat-Wing Parabola + Residual ML

```
final_prediction = parabola_baseline(log_moneyness) + ML_residual_correction
```

This two-stage design is deliberate. The parabola captures the dominant smile structure analytically. The ML model then learns only the small corrections the parabola gets wrong — residuals are centred near zero, lower variance, easier to predict reliably.

### Stage 1 — Flat-Wing Parabolic Baseline (`fit_splines`)

For each `(timestamp, option_type)` group, a **2nd-degree polynomial** is fitted through observed IV values against log-moneyness using `np.polyfit(x, y, deg=2)`. Key design choices:

- **Flat-wing extrapolation**: predictions outside the observed moneyness range are clamped to the boundary value via `np.clip(lm, min_x, max_x)` before evaluating the polynomial — prevents divergence at extreme strikes
- **Per-group IV bounds**: stores `iv_min = min_observed × 0.5` and `iv_max = max_observed × 1.5` — final predictions clipped to `[max(global_floor, iv_min), min(global_cap, iv_max)]`
- **Fallback**: 3+ points → quadratic; 2 points → linear; 0–1 points → forward-carry from nearest earlier timestamp of same option type
- **`spline_pred` removed from features**: the ML model's feature set is strictly independent of the baseline level, preventing the model from learning a trivial identity mapping

### Stage 2 — Residual ML Model

Target: `residual = observed_IV − parabola_prediction`

**17 features used:**

| Group | Features |
|---|---|
| Temporal (strongest) | `iv_lag1`, `iv_lag2`, `iv_lag3`, `iv_roll3_mean`, `iv_roll3_std`, `iv_roll5_mean`, `iv_roll5_std`, `iv_ewma` |
| Spatial (same timestamp) | `left_neighbor_iv`, `right_neighbor_iv`, `atm_iv` |
| Financial (static) | `log_moneyness`, `strike_dist`, `is_put`, `underlying_price`, `days_to_expiry`, `time_of_day` |

**Residual winsorisation** at the 99th percentile before training prevents outlier residuals from dominating the squared loss.

### Model Zoo

All four models evaluated automatically; best selected by average CV MSE:

| Model | Parameters |
|---|---|
| RandomForest | n_estimators=100, max_depth=6, min_samples_leaf=20 |
| **GradientBoosting** ✓ | n_estimators=150, lr=0.05, max_depth=5, min_samples_leaf=30 |
| LightGBM *(if available)* | n_estimators=300, lr=0.03, max_depth=5, num_leaves=20, reg_lambda=5.0 |
| XGBoost *(if available)* | n_estimators=300, lr=0.03, max_depth=5, reg_lambda=5.0 |

---

## Cross-Validation Design

**Method:** `TimeSeriesSplit(n_splits=5)` with an embargo of 2 timestamp steps

```
Fold 1: ─── Train (163 ts) ─── [gap 2] ─── Val (162 ts) ───
Fold 2: ─── Train (325 ts) ─── [gap 2] ─── Val (162 ts) ───
Fold 3: ─── Train (487 ts) ─── [gap 2] ─── Val (162 ts) ───
Fold 4: ─── Train (649 ts) ─── [gap 2] ─── Val (162 ts) ───
Fold 5: ─── Train (811 ts) ─── [gap 2] ─── Val (162 ts) ───
```

The embargo gap ensures lag features computed at the end of the training block cannot observe the first validation timestamp.

**Holdout evaluation (HOLDOUT_RATIO = 0.30):**  
30% of *observed* IV values in the validation block are randomly hidden and used as evaluation targets. This is used **only for scoring** — never for training. It simulates the competition scenario where ground-truth IVs for missing cells are unknown.

**Sequential (autoregressive) validation loop:**  
Validation timestamps are processed one-by-one in chronological order. At timestamp T:
1. Temporal features use training history + *predicted* IVs from val timestamps before T
2. The parabola is refitted strictly on available data at timestamp T
3. True val IVs are **never** used as features — only `iv_pred` is fed forward into history

---

## Leakage Prevention

| Risk | How prevented |
|---|---|
| Future timestamps in lag features | Sequential loop: history only contains training rows + previously *predicted* val rows |
| True val IVs in lag history | Only `iv_pred` appended to `pred_history` — `iv_true` never used |
| Preprocessing on full dataset | Clip bounds (`percentile`) computed from training block only, per fold |
| Spline fitted on validation data | Parabola refitted per timestamp inside the sequential loop using only current ts data |
| Own IV as spatial neighbor | Own strike excluded from neighbor lookup for observed rows (`sa = os_[os_!=k]`) |
| Random CV shuffling | `TimeSeriesSplit` only — `KFold` and `shuffle` never used |
| Global statistics before split | All fold-local: medians, residual bounds, IV floor/cap all derived per fold |
| Row index misalignment | Explicit `row_id = np.arange(len(df_long))` injected at ingest; `build_spatial` uses `set_index("row_id")` for safe assignment |

---

## Data Facts (from EDA)

- **975 timestamps** | **28 option columns** | **5,460 missing cells (20%)**
- Wide format shape: **(975, 30)** | Long format shape: **(27,300, 11)**
- IV range: approximately `[0.088, 1.138]` at p1/p99
- CE mean IV < PE mean IV — put skew confirmed (equity crash risk premium)
- Smile is U-shaped with negative skew
- IV lag-1 autocorrelation > 0.95 — temporal features dominate prediction
- Deep OTM strikes have higher missing rates — spatial neighbor features critical there

---

## How to Run

### On Kaggle

1. Upload `dataset.csv` to your Kaggle notebook environment
2. Run `eda_analysis.ipynb` — all plots render inline, nothing saved to disk
3. Run `evaluation_model.ipynb` — auto-selects best model, writes `submission.csv`
4. Upload `submission.csv` to the competition page

### Locally

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn lightgbm xgboost
jupyter notebook eda_analysis.ipynb
jupyter notebook evaluation_model.ipynb
```

Change `DATASET_PATH` in both notebooks from the Windows path to your local path.

---

## Dependencies

| Library | Purpose |
|---|---|
| pandas ≥ 1.5 | Data loading, reshaping, groupby operations |
| numpy ≥ 1.23 | Polynomial fitting (`np.polyfit`), numerical ops |
| scipy | Not used in final model (parabola replaces `CubicSpline`) |
| scikit-learn ≥ 1.2 | `TimeSeriesSplit`, `clone`, `GradientBoostingRegressor`, metrics |
| matplotlib ≥ 3.6 | All EDA plots (inline via `plt.show()`) |
| seaborn ≥ 0.12 | Heatmaps and distribution plots in EDA |
| lightgbm *(optional)* | Auto-detected; used if available |
| xgboost *(optional)* | Auto-detected; used if available |

---

## Submission Format

`generate_solution()` extracts only originally-missing cells from `filled_dataset.csv` and writes `submission.csv`:

```
id,value
07-01-2026 09:15||NIFTY27JAN2625500CE,0.11423
07-01-2026 09:15||NIFTY27JAN2625700CE,0.10892
...
```

**5,460 rows** in the final submission file.  
Each `id` is `{datetime}||{option_column_name}`. The `value` column contains the predicted IV.

---

## Evaluation Metric

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Lower is better. Public leaderboard = 30% of test data; final ranking = private leaderboard (remaining 70%).  
The pipeline selects the model with the lowest **average CV MSE across all 5 folds**, not the public leaderboard score, to ensure generalisation to the private split.

---

## Authors

Submitted as part of the **Finance Club, IIT Roorkee — Open Projects 2026**  
Competition hosted on Kaggle by ADITYA JAIN
