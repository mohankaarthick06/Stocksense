# WORKLOG — Person 1 (Data Analyst)

A running record of what was done, in order, and why. Newest entries at the bottom.
Scope (from the brief, section 10): cleaning, integration, EDA, statistics, data quality.

## Output map
| Output | Location |
|---|---|
| Cleaning + master-table pipeline | `src/data_prep.py` |
| EDA + statistics code | `src/eda_stats.py` |
| Cleaned master dataset | `data/processed/master_dataset.csv` |
| Data Quality Report | `reports/data_quality_report.md` / `.csv` |
| Statistical tests | `reports/statistical_tests.md` / `.csv` |
| EDA charts | `reports/figures/` |
| Notebook | `notebooks/01_cleaning_eda_stats.ipynb` |

## Log

### Step 0 — Understand the brief and inspect the data
- Read the PDF end to end and listed every Person 1 requirement (cleaning, integration, DQ report, 6 EDA questions, 3+ statistical tests).
- Read `data/README.md` and `DECISIONS.md`. Treated `DECISIONS.md` as a hypothesis list and checked its claims against the data.
- Profiled all 8 raw files. Issue counts found (before cleaning): 5,742 duplicate transactions, 8,683 `DD-MM-YYYY` dates,
  ~2,890 malformed IDs, 2,165 quantities <= 0, 45 keying errors (x100), 3,620 missing prices, 715 prices ~10x MRP,
  208 duplicate inventory rows, 919 inventory arithmetic mismatches, 40 negative closing stocks, 277 missing inventory days,
  city alias `Trichy`, 12 impossible temperatures, 28 missing temperatures, 19 missing rain values, 3 duplicate product rows.
- The answer key is not in the repo, so I cannot score against it.

### Step 1 — Deeper checks before choosing cleaning rules
Each rule below is based on evidence from the data, not on an assumption.
- **Inventory mismatches (919)** split into two exact patterns, using the next day's `opening_stock` (the physical count):
  - 709 rows: next opening = `opening + received - sold` → the **closing count was mistyped**. Fix: recompute closing.
  - 208 rows: next opening = recorded closing → the count is right but a **delivery was not recorded** (`received_qty = 0`).
    All 208 match a purchase-order receipt of exactly the missing amount. Fix: take `received_qty` from purchase orders.
  - 2 rows have no next day to check against; the same two rules resolve them (PO receipt first, then arithmetic).
  - All 40 negative closing stocks are among the mistyped closings.
- **Missing inventory days (277)**: 275 single-day gaps and one 2-day gap. Rebuild each from the previous closing, PO receipts and
  cleaned transactions, then check that the rebuilt closing equals the next day's opening.
- **Prices**: within one store × product × date × discount, every valid line has the **same** price (66,632 of 66,632 groups).
  So a missing price can be recovered exactly from its sibling lines. The 715 prices about 10× MRP divided by 10 give valid prices.
- **Promotions**: the calendar and the transaction `promotion_flag` agree on 100% of store-product-days, with identical discounts.
  So the master uses the calendar, which also covers days with zero sales.
- **External factors**: weekend flags are all correct. Holidays and festivals are the same in every city. 12 impossible temperatures (-1, 0, 3.2, 58.4, 99.9, 112 °C).
- **Transactions vs inventory**: after removing returns and duplicates, daily sums equal `inventory.sold_qty` on every row
  except 45. Those 45 are exactly the x100 keying errors, so inventory is the reliable source there.

### Step 2 — Cleaning and integration pipeline (`src/data_prep.py`)
- One function per table. Every fix goes through `DQLog.add(...)` → `reports/data_quality_log.csv`, so the report is generated, not hand-typed.
- Order matters: products → stores → weather → transactions → purchase orders → promotions → inventory → master.
  Inventory comes after transactions and POs because it is rebuilt from both.
- Keying errors: **corrected** (÷100) rather than dropped, because inventory confirms the real quantity. Dropping would lose a genuine sale.
- Master grid = every store × product from its first inventory day to 31 Aug. Sales are left-joined and zero-sale days get 0,
  otherwise average demand would be overstated and stock-out days would disappear.
- **Stock-out definition:** `stockout_flag = 1` when the reconciled `closing_stock == 0` at end of day. It is objective and is what a manager sees.
  `below_reorder_flag` (closing < reorder level) is provided as a secondary "at risk" signal.
- **Censored demand:** on stock-out days sales understate demand. `demand_est = max(sold, typical)`, where typical is the
  mean of the last 4 same-weekday non-stock-out days (fallback: last 7 days), **past data only**, so there is no leakage.
  `lost_units_est = demand_est − units_sold`, and `lost_sales_est = lost_units × unit_price`.
- Assertions guarantee:
  - stock arithmetic holds on every row, and no stock is negative;
  - each day's opening = the previous day's closing;
  - inventory sold = transactions sold on every day;
  - the row count is unchanged after all 7 joins, with 0 unmatched keys;
  - there is exactly one row per date × store × product.

**Run 1 result:** 18 s. 69,554 rows × 57 columns. All assertions pass. Final counts differ slightly from the Step 0 profile because they are counted *after* de-duplication (e.g. 8,612 DD-MM-YYYY dates instead of 8,683).

**Sanity check against the data README:** stock-out rate 6.31% (README 6.3%). On promo vs non-promo days, 13.6% vs 5.2% (same as README).
Beverages 18.1% (README 18%). Worst stores S05 and S02 (as README). ✔
Estimated lost demand is 4.1% vs about 6% true, so the estimate is **conservative**: once the shelf is empty, extra demand cannot be observed.
Fix after run 1: `promo_type` "None" was being read back as NaN by pandas → renamed to "No Promo".

### Step 3 — KPIs, EDA and statistics (`src/eda_stats.py`)
- **KPIs** (brief section 1): revenue Rs 7.19 Cr, 1.05 M units, stock-out rate 6.3%, days of inventory 3.6,
  estimated lost sales Rs 25.6 L (3.6% of revenue). Overall and per store → `reports/kpi_summary.csv`.
- **10 charts**, one business question each. All 6 required by the brief, plus 4 extra: supplier reliability, the reorder reset, lost sales, and heat vs beverages.
  Insight text is generated from the numbers, so it stays correct if the data changes.
- **7 statistical tests** (3 required + 4 supporting). Each has business question, H0, H1, test, *why this test*,
  p-value, **effect size**, and a manager action. Tests are paired or rank-based because demand is skewed and product mix confounds pooled comparisons.
- Charts reviewed visually. Fixed: gridlines drawn over bars, colliding line labels, overlapping supplier labels, and truncated city names.

**Corrections made after checking the data (important for the pitch):**
1. *Pareto*: the planned wording "revenue is concentrated" was wrong. It takes 42 of 81 products (52%) to reach 80%.
   Text changed to: not a classic 80/20 pattern; the top 10 products give 37%.
2. *Censored-demand story*: `DECISIONS.md` / README said the 1 June reset **cut** levels for products that were stocking out. The data shows
   the reset used one rule for all (~3.7 days of May *sales*). Levels for May-stock-out products actually **rose 23%**, but because their
   May sales were capped they got only 3.4 days of estimated *demand* vs 3.7 for others. Their stock-outs fell from 9.0% to 6.0% but stay
   2.3× the rest (2.6%). The chart and T6 now say "under-corrected", not "cut". May was excluded from the before/after comparison because it defines the groups.

**Test results:**
| # | Question | Result |
|---|---|---|
| T1 | Promotions increase sales? | Yes: median +29% (paired), significant in 7/7 categories |
| T2 | Demand differs by store type? | Yes: Hypermarket 36.6 vs Supermarket 10.4 vs Express 5.7 units/day for the same products (epsilon² 0.42) |
| T3 | Stock-out associated with promotion? | Yes: 13.6% vs 5.2%, OR 2.9 (3.0 adjusted for store × category) |
| T4 | Temperature drives beverages? | Yes in 6/6 stores, rho 0.32–0.53 |
| T5 | Late delivery depends on supplier? | Yes: SUP04 36% late vs 7%, OR 7.7 |
| T6 | May stock-outs persist after the reset? | Yes: 6.0% vs 2.6% |
| T7 | Weekend differs from weekday? | Yes, by format: Hypermarket +45%, Supermarket +23%, Express −6% |

### Step 4 — Notebook, reports, reproducibility
- `notebooks/01_cleaning_eda_stats.ipynb` calls the same `src/` functions (one source of truth), executed with outputs saved
  (10 charts, 0 errors). It ends with a hand-off section for Student 2 and Student 3.
- Generated reports: `data_quality_report.md` (row counts, 30 logged issues, integrity checks, definitions, limitations, column dictionary),
  `eda_insights.md`, `statistical_tests.md`, plus CSV versions.
- `requirements.txt` added for Round 1.
- **Reproducibility check:** deleted `data/processed/` and `reports/`, then re-ran everything from raw. All outputs were regenerated identically, all assertions pass.

### Step 5 — Review: "is Person 1's work correct?"
**A. Independent verification.** A separate script, which does *not* import `src/`, recomputed the key numbers from the raw files with different code:
- Grain: one row per date × store × product; 387 combinations; 1 Mar – 31 Aug; no missing days. ✔
- Units sold match an independent re-clean on **every** store-product-day (0 differing rows). Revenue matches to the rupee (Rs 71,930,732). ✔
- Stock arithmetic holds on every row; opening = previous closing; no negative stock; never sold more than available;
  `received_qty` = purchase-order receipts on every row. ✔
- Joins: S05 has weather (Trichy fix works); S02 and S06 share Chennai weather; 7 categories; 3 store types;
  master promo flag = transaction promo flag on every sale day. ✔
- `stockout_flag` = (closing == 0); `demand_est` ≥ sales and = sales on normal days. On S05-Cola (68 stock-out days),
  `demand_est` was re-derived from past-only data. The first attempt had 1 mismatch, caused by the *verifier* using a different fallback rule;
  the pipeline follows its documented rule. ✔
- Headline rates match the data README (6.3%, 13.6% / 5.2%, 18.1%). The 3 mandatory tests reproduce with a simpler setup
  (e.g. T1 without the minimum-days filter: median lift +28.9%, p ≈ 1e-57). ✔

**B. Gap found against the brief → fixed.** The brief says *"Audit data types, missing values, duplicates and invalid values"*.
The report only listed columns that had problems, and checks that passed (hour range, payment modes, discount/promo
consistency, PO dates…) were never written down. Added `audit_raw_files()`: **62 explicit checks covering every raw column**
(type as read, expected type, missing, rule, violations). The result is 41 clean and 21 with issues, all fixed. The audit is in the DQ report,
`reports/raw_column_audit.csv` and the notebook.

**C. New finding from the audit.** 9 purchase orders are marked RECEIVED with **0 units** delivered. Inventory confirms nothing
arrived, so these are real failed deliveries, not typos. They are kept (fill rate 0) and logged.

**D. Bugs in my own audit code, caught and fixed.**
1. The price rule used `between(0.01, None)`, which is False for every row → replaced with `> 0`.
2. The raw price violations (4,340) were reconciled exactly: 3,620 missing + 720 about 10× MRP (715 after removing 5 duplicate rows).

**E. Limitation now quantified.** Of 4,391 stock-out days, `demand_est` raised 2,920 above sales. On 1,470 the shelf emptied
after sales had already reached the typical level, so no loss is assumed. Only 1 day had no clean history at all. This is why
lost sales (Rs 25.6 L) is a lower bound.

**Checklist against the brief (Round 1 / Student 1):**
| Brief requirement | Status | Where |
|---|---|---|
| Audit types, missing, duplicates, invalid values | ✔ (added in Step 5) | DQ report → Column audit |
| Standardise categories, dates, IDs | ✔ | DQ log rows (categories, dates, IDs, store_type, city) |
| Validate inventory arithmetic | ✔ | 919 mismatches diagnosed; asserted on every row |
| Aggregate transactions to daily store × product | ✔ | `build_master` |
| Merge product, store, inventory, external | ✔ (+ promotions, POs) | row count unchanged after 7 joins |
| DQ report: issue, count, action, justification | ✔ | `reports/data_quality_report.md` |
| Traps: missing, duplicates, categories, impossible qty, inventory mismatch, sparse history | ✔ all 6 | DQ log |
| EDA: 6 business questions | ✔ 6 + 4 extra | `reports/eda_insights.md` |
| ≥ 3 stats tests with question, H0, H1, test, p, interpretation | ✔ 7 tests | `reports/statistical_tests.md` |
| KPIs from the brief | ✔ all 7 + extras | `reports/kpi_summary.csv` |
| Cleaned master CSV, notebook | ✔ | `data/processed/`, `notebooks/` |

## How to reproduce
```
pip install -r requirements.txt
python src/data_prep.py      # ~20 s  -> data/processed/*, reports/data_quality_*
python src/eda_stats.py      # ~15 s  -> reports/figures/*, eda_insights.md, statistical_tests.md, kpi_summary.csv
jupyter notebook notebooks/01_cleaning_eda_stats.ipynb
```

## Open items / notes for the team
- The answer key (`data/answer_key_DO_NOT_COMMIT/`) is not in the repo, so the numbers cannot yet be scored against the truth.
- `transactions_clean.csv` is 46 MB (under GitHub's 100 MB limit). Decide whether to commit it or regenerate it from the script.
- `README.md` shows as deleted in git status. That was not done in this session; the team README is still to be written.

---

# WORKLOG — Person 2 (ML Engineer)

Scope (from the brief, section 10): feature engineering, forecasting (Model 1), validation.
Input: Round 1's `data/processed/` tables. Output: the modelling table for both models, the demand model, and forecasts for Student 3.

## Output map
| Output | Location |
|---|---|
| Feature engineering + leakage proof | `src/features.py` → `data/processed/features.csv` |
| Model 1 training, tuning, evaluation | `src/train_forecast.py` |
| Load / predict helpers for dashboard | `src/forecast.py` |
| Saved model | `models/demand_forecast.joblib` (+ `demand_forecast_meta.json`) |
| Forecast 1-7 Sep | `data/processed/forecast_next7.csv` |
| Out-of-time forecasts (Model 2 input) | `data/processed/forecast_oot.csv` |
| Forecast error profiles (safety stock) | `reports/forecast_error_by_category.csv`, `data/processed/forecast_error_by_combo.csv` |
| Reports | `reports/model1_demand_forecast.md`, `reports/feature_dictionary.md`, `reports/model1_comparison.csv`, `reports/model1_tuning_log.csv` |
| Charts | `reports/figures/11_...` to `16_...` |
| Notebook | `notebooks/02_features_forecasting.ipynb` (rebuilt by `notebooks/build_02_notebook.py`) |

## Log

### Step 0 — Environment
- Anaconda's scikit-learn was compiled for NumPy 1.x, but NumPy 2.5 is installed, so `import sklearn` fails. The fix is a
  project-local `.venv` (git-ignored) with current pandas / scikit-learn / xgboost. The global install is not touched.
  Round 1 scripts also run in it.

### Step 1 — The modelling table (`src/features.py`)
- **One row = one forecast origin**: close of day *t* for one store x product. Both targets are on the same row, so Model 1
  and Model 2 describe the same week:
  - `next_7_day_demand` = sum of `demand_est` over *t+1..t+7*. Uses `demand_est`, not raw sales, as Round 1 handed off.
  - `stockout_next7` = any stock-out in *t+1..t+7*. It is 28.2% positive, matching Round 1's figure.
- **61 features in 8 groups:**
  - recent demand;
  - promotion plan (from the calendar, never from future transactions);
  - festival / holiday / payday / local-event calendar;
  - past weather;
  - price (MRP hikes);
  - stock & supply (open POs counted by *expected* date, expanding supplier late rate);
  - store / product profile;
  - calendar position.
- Dropped `weekend_days_next7`: every 7-day window has exactly 2 weekend days, so it is a constant.
- **Leakage proof** (`check_no_leakage`): for 4 origin dates, all data after *t* is deleted (sales, stock, weather, PO arrivals),
  the features are rebuilt, and all 61 must be identical. Result: 1,516 rows, all identical.
  It caught 2 real leaks while building, both fixed:
  1. The product "festival lift" was first learned over the whole period. It is now an expanding estimate from festivals already past.
  2. Festival dates were read from the master table, which stops at *t*. They now come from the published calendar, which runs to 7 Sep.
- Spot check: S02 Cola on 10 Jun shows a stock-out that day, 768 units on order from SUP04 (34% late), a post-promotion dip, and
  +9% price change from the 1 Jun MRP hike. All of these are consistent with Round 1's findings.

### Step 2 — Split, baseline, algorithms (`src/train_forecast.py`)
- **Time-aware split with 7-day gaps** (from DECISIONS D2.4):

  | Split | Origins | Rows |
  |---|---|---|
  | Train | 8 Mar – 12 Jul | 47,879 |
  | Validation | 20 Jul – 3 Aug | 5,655 |
  | Test | 11 – 24 Aug | 5,390 |

  A training origin on 12 Jul has a target ending 19 Jul, so validation cannot sit inside a training label. Asserted in code.
- **Baseline first**: 7 x mean demand of the last 7 days.
- **All permitted regression algorithms, 42 configurations, tuned on validation only**: Linear Regression, Decision Tree, Random Forest,
  XGBoost (squared error / tweedie / log target, early stopping), KNN, SVM (linear kernel, and RBF kernel on a 12k-row subsample
  because its training time grows quadratically with rows). Naive Bayes is a classifier, so it does not apply.
- Distance and linear models get one-hot categories, a median imputer with missing-value flags, signed log1p and scaling, and a log target.
  Tree models use native NaN handling.
- The best configuration per algorithm is refitted on train + validation, and the **test set is scored once**.

### Step 3 — Results
| | Validation WAPE | Test WAPE | Test bias |
|---|---|---|---|
| Baseline | 22.7% | 20.8% | −5.8% |
| **XGBoost (selected)** | 14.9% | **15.3%** | −4.5% |
| Random Forest | 15.7% | 15.8% | −1.5% |
| SVM (linear) | 15.3% | 15.8% | −7.0% |
| Linear Regression | 15.3% | 15.9% | −7.0% |
| Decision Tree | 17.1% | 17.2% | −3.7% |
| KNN | 21.2% | 19.3% | −8.5% |

- **Selected: XGBoost**, squared error on log target, depth 5, 727 trees. It has 26% less error than the baseline, and the ranking held on test.
- **Metric choice:** selected on WAPE, with bias as the second check. MAPE explodes on slow movers, and RMSE and R² are dominated by the largest items.
- **Honest note:** XGBoost's margin over RF / SVM / LR is only about 0.5 WAPE points. Most of the gain comes from the features, which all models share.
- Weakest segments: Frozen and Snacks (about 20%, promotion- and heat-driven). Dairy is the most accurate at 12% but gains least over the baseline
  (dairy demand is stable). Promotion weeks are the hardest at 19% WAPE.
- **Permutation importance (test):** recent demand dominates. After that come stock & supply (`reorder_level` is the planners' own demand estimate),
  the store/product profile, and the promotion plan.

### Step 4 — A selection rule I tried and dropped (disclosed on purpose)
- All models under-forecast slightly. I tried a tie-break: among configurations within 0.25 WAPE points of the best, pick the least biased.
- Across algorithms, this picked the linear SVM, because its **validation** bias was −0.8%. On test the same model's bias was **−7.0%**,
  and its WAPE was 15.9% against 15.3%.
- Conclusion: bias measured on one 15-day validation window is noise, not a property of the model. I reverted to the rule fixed in
  DECISIONS.md before training: lowest validation WAPE.
- **Disclosure:** the revert was decided after the test results had been seen. It restores the rule that was planned in advance;
  it does not pick the test winner.
- Bias is handled downstream instead. The out-of-time error profile (`bias_units`, `sigma_7d` per category and per item) goes to the safety-stock formula.

### Step 5 — Outputs for Student 3
- **Final model:** refitted on all labelled origins (8 Mar – 24 Aug) and saved with its feature list and category encoding.
  `src/forecast.py` `load_model()` / `predict()` re-predict the saved forecast exactly (checked in the notebook).
- **Forecast for 1-7 Sep** (origin 31 Aug): 387 rows, with `forecast_sigma_7d` and an 80% range (p10 / p90).
  Sigma comes from each item's out-of-time errors, falling back to its category's relative error when an item has less than 28 rows.
- **Out-of-time forecasts** for every origin from 12 Apr. Each 14-day block is predicted by a model trained only on origins whose targets
  ended before the block, so Model 2 can use `coverage = (stock + incoming) / forecast` without leakage (DECISIONS D2.7).

### Step 6 — Scored against the answer key (evaluation only)
The answer key is git-ignored and was never used for training, tuning or selection.
- **Correcting for empty shelves matters.** With the same model and settings, trained on raw `units_sold` it under-forecasts true demand by
  9.4% (WAPE 18.4%). Trained on `demand_est` the under-forecast is 5.8% (WAPE 16.4%). The Round 1 correction is worth about 2 WAPE points.
- **Unseen week 1-7 Sep:** forecast WAPE is **14.7%** vs 19.3% for the baseline, with bias −2.4%.

### Checklist against the brief (Round 2 / Student 2)
| Brief requirement | Status | Where |
|---|---|---|
| Time features (day_of_week, weekend, month, week, festival) | ✔ (window-based: holidays / festivals / paydays in *t+1..t+7*) | `features.py` |
| Lag 1 / 7 / 14 | ✔ | `features.py` |
| Rolling mean 7 / 14, std 7 | ✔ + 28-day mean/std, CV, trend | `features.py` |
| Inventory: days of inventory, inventory-to-demand, reorder gap | ✔ | `features.py` |
| Price / promo: discount, price change, promotion flag | ✔ (from the planned calendar) | `features.py` |
| Store / product: store type, category, shelf life, lead time | ✔ (brand left out: it maps almost 1-to-1 onto sub-category, which is used) | `features.py` |
| Avoid leakage | ✔ proven by rebuild test | `check_no_leakage()` |
| Baseline first (previous 7-day mean) | ✔ | report |
| Compare permitted regression algorithms | ✔ all 6 applicable | `model1_comparison.csv` |
| Time-aware train / validation / test | ✔ with gaps | report |
| MAE, RMSE, MAPE, R²; which metric matters | ✔ + WAPE, bias | report |
| Saved model / reproducible pipeline | ✔ | `models/`, notebook |

## How to reproduce (Round 2)
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python src/features.py         # ~50 s  -> data/processed/features.csv, reports/feature_dictionary.*
.venv\Scripts\python src/train_forecast.py   # ~7 min -> models/, forecasts, reports/model1_*, figures 11-16
```


---

# WORKLOG -- Student 3 (Decision Intelligence)

Scope: Model 2 (stock-out risk classifier), explainability, business action layer, Streamlit dashboard, reports.

## Output map

| Output | Location |
|---|---|
| Stock-out classifier training | src/train_stockout.py |
| Stockout load/predict interface | src/stockout.py |
| Explainability | src/explain.py |
| Business action layer | src/recommend.py |
| Sep 1-7 recommendation pipeline | src/make_recommendations.py |
| Streamlit dashboard | dashboard/app.py |
| Unit tests (9, all pass) | 	ests/test_recommend.py |
| Sep 1-7 action table | data/processed/recommendations_1_7_sep.csv |
| Model bundle | models/stockout_model.joblib |
| Round 3 summary report | 
eports/round3_decision_intelligence.md |
| Figures 17-21 | 
eports/figures/ |

## Key decisions

- Algorithm: LinearSVM(C=1.0) selected via val PR-AUC (0.6134 > XGB 0.6104).
- Calibration: isotonic regression (Brier 0.166 -> 0.137 on test).
- Imbalance: class_weight=balanced; SMOTE excluded (time series).
- KNN/SVM run on 12k subsample for speed (stated in code with rationale).
- Bias: adjusted_forecast = forecast_7d - bias_units (positive bias = over-forecast).
- Stacked features from forecast_oot (no leakage): coverage, forecast_vs_rm7, lead_cover_gap.

## Results

- Test PR-AUC: 0.6299 (rule baseline 0.3133 -- 2x improvement)
- Test ROC-AUC: 0.7988, Recall@0.40: 0.499, Brier: 0.137
- Sep 1-7: 62 HIGH, 122 MEDIUM, 203 LOW risk items
- Expected loss at risk: Rs 4,20,336

## Checklist

| Task | Status |
|---|---|
| Model 2 classifier + calibration | DONE |
| All algorithm comparison (NB, DT, RF, XGB, KNN, SVM) | DONE |
| Global explainability (permutation importance) | DONE |
| Local what-if flip | DONE |
| Safety stock / reorder qty / risk level | DONE |
| Manager action matrix | DONE |
| Expected loss (Rs) | DONE |
| Sep 1-7 action table (387 rows) | DONE |
| 6-page Streamlit dashboard | DONE |
| What-If simulator | DONE |
| Unit tests (9/9 pass) | DONE |
| Round 3 report | DONE |
| README | DONE |

## How to reproduce

    python src/train_stockout.py          # 8 min -> models/stockout_model.joblib, reports/model2_*
    python src/explain.py                 # 2 min -> reports/model2_feature_importance*, fig 21
    python src/make_recommendations.py   -> data/processed/recommendations_1_7_sep.csv
    python tests/test_recommend.py        -> 9/9 tests pass
    streamlit run dashboard/app.py        -> http://localhost:8501
    python scripts/generate_reports.py   -> reports/round3_decision_intelligence.md
