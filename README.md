# StockSense — IntelliData 2026 (NovaMart Retail)

> **Team project.** This README covers all three rounds and the complete repo structure.

---

## Project Overview

StockSense is a data-science decision intelligence system for NovaMart, a multi-format Indian retail chain with stores in Bengaluru, Mumbai, and Delhi. The system forecasts 7-day demand, classifies stock-out risk, and generates actionable replenishment orders for every SKU-store combination.

**Competition:** IntelliData 2026 — NovaMart Retail challenge  
**Period:** Mar 12 – Aug 24, 2026 (historical); Sep 1–7 forecast horizon  
**Data:** 3 stores × 43 products × 166 days ≈ 21,000 daily records

---

## Round Summary

| Round | Focus | Primary outputs |
|---|---|---|
| 1 | Data cleaning + EDA | `data/processed/master_dataset.csv`, EDA notebooks |
| 2 | Feature engineering + Model 1 (demand forecast) | `models/demand_model.joblib`, `data/processed/forecast_next7.csv` |
| 3 | Model 2 (stock-out risk) + explainability + business actions + dashboard | See below |

---

## Round 3 — Decision Intelligence (Student 3)

### What's new in Round 3

| Component | Description |
|---|---|
| **Model 2** | Stock-out risk classifier (calibrated LinearSVM, PR-AUC 0.63, 2× rule baseline) |
| **Explainability** | Global permutation importance + local what-if flip per item |
| **Business action layer** | Safety stock, reorder qty, risk level, manager action, expected loss |
| **Recommendation table** | 387 items for Sep 1–7 with ₹4.2 L total expected loss flagged |
| **Streamlit dashboard** | 6 pages + What-If simulator |

### Running Round 3 end-to-end

```bash
# 1. Train stock-out classifier (saves models/stockout_model.joblib)
python src/train_stockout.py

# 2. Global explainability (saves figures 17-21, feature importance CSVs)
python src/explain.py

# 3. Business actions (saves data/processed/recommendations_1_7_sep.csv)
python src/make_recommendations.py

# 4. Launch dashboard
streamlit run dashboard/app.py

# 5. Generate final report
python scripts/generate_reports.py

# 6. Unit tests
python tests/test_recommend.py
```

### Dashboard pages

| Page | What it shows |
|---|---|
| 📊 Executive Summary | KPI cards, revenue trend, risk donut, top stores |
| 📈 Demand Intelligence | Actual vs forecast per item, error by category, weekly demand |
| ⚠️ Inventory Risk | Store × category heatmap, risk table, sortable filters |
| 🎯 Manager Action Centre | Ranked action cards, What-If simulator, CSV download |
| 🤖 Model Performance | Model 1 & 2 comparison tables, evaluation charts |
| 🔍 Explainability | Group importance chart, per-item driver sentences |

---

## Repository Structure

```
StockSense-main/
├── data/
│   ├── raw/                          # Original data (DO NOT MODIFY)
│   └── processed/
│       ├── master_dataset.csv        # Round 1: cleaned
│       ├── features.csv              # Round 2: engineered features
│       ├── forecast_oot.csv          # Round 2: OOT forecast (leakage-free)
│       ├── forecast_next7.csv        # Round 2: Sep 1-7 forecast
│       ├── forecast_error_by_combo.csv
│       └── recommendations_1_7_sep.csv  # Round 3: action table
├── models/
│   ├── demand_model.joblib           # Round 2: Model 1
│   ├── stockout_model.joblib         # Round 3: Model 2 (calibrated)
│   └── stockout_model_meta.json
├── notebooks/
│   ├── 01_EDA.ipynb                  # Round 1
│   ├── 02_Features.ipynb             # Round 2
│   └── 03_Modelling.ipynb            # Round 2
├── src/
│   ├── data_prep.py                  # Round 1 (do not edit)
│   ├── eda_stats.py                  # Round 1 (do not edit)
│   ├── features.py                   # Round 2 (do not edit)
│   ├── train_forecast.py             # Round 2 (do not edit)
│   ├── forecast.py                   # Round 2 (do not edit)
│   ├── train_stockout.py             # Round 3 ← Model 2 training
│   ├── stockout.py                   # Round 3 ← Load/predict helpers
│   ├── explain.py                    # Round 3 ← Explainability
│   ├── recommend.py                  # Round 3 ← Business action layer
│   └── make_recommendations.py      # Round 3 ← Sep 1-7 action table
├── dashboard/
│   ├── app.py                        # Round 3 ← Streamlit dashboard
│   ├── data_access.py                # Round 3 ← Cached loaders
│   └── components.py                 # Round 3 ← Plotly charts & cards
├── reports/
│   ├── figures/
│   │   ├── 17_calibration_curve.png
│   │   ├── 18_roc_pr_curves.png
│   │   ├── 19_confusion_matrices.png
│   │   ├── 20_model2_comparison.png
│   │   └── 21_model2_permutation_importance.png
│   ├── model2_tuning_log.csv
│   ├── model2_comparison.csv
│   ├── model2_feature_importance.csv
│   ├── model2_feature_importance_groups.csv
│   ├── model2_stockout_risk.md
│   └── round3_decision_intelligence.md
├── tests/
│   └── test_recommend.py             # 9 unit tests (all pass)
├── scripts/
│   └── generate_reports.py
├── DECISIONS.md                      # Project-wide design decisions log
└── WORKLOG.md                        # Per-student task log
```

---

## Model 2 — Stock-out Risk Classifier

### Design

- **Target:** `stockout_next7` — will this SKU-store stock out in the next 7 days?
- **Positive rate:** 26.4% (manageable imbalance; no SMOTE — time-series structure)
- **Algorithm selection metric:** PR-AUC on validation set (recall priority; accuracy misleads with 28% positives)
- **Calibration:** isotonic regression wrapper improves Brier score 0.166 → 0.137

### Train / Val / Test splits

| Split | Dates |
|---|---|
| Train | Apr 12 – Jul 12, 2026 |
| Val | Jul 20 – Aug 3, 2026 |
| Test | Aug 11 – Aug 24, 2026 |

### Results (test set)

| Metric | Score |
|---|---|
| PR-AUC | **0.6299** |
| ROC-AUC | 0.7988 |
| Recall@0.40 | 0.4989 |
| Brier | 0.1372 |
| Rule baseline PR-AUC | 0.3133 |

### Algorithm rules

Machine learning only. Permitted: Linear Regression, Decision Tree, SVM, XGBoost, KNN, Naive Bayes, Random Forest. Logistic Regression is **not permitted** (DECISIONS D2.7). Calibration via `CalibratedClassifierCV` (sklearn wrapper) is allowed.

---

## Business Action Layer

| Formula | Notes |
|---|---|
| `adjusted_forecast = forecast_7d − bias_units` | Positive bias = model over-forecasts; subtract to unbias |
| `safety_stock = z × σ₇d × √((effective_lead + 1) / 7)` | z=1.65 non-perishable; z=1.28 shelf_life < 7 days |
| `reorder_qty = max(0, recommended_stock − stock − incoming)` | Rounded up to integer |
| `risk_level` | HIGH ≥ 0.70; MEDIUM ≥ 0.40; LOW < 0.40 |
| `expected_loss = prob × max(0, forecast − stock − incoming) × price` | Revenue at risk |

---

## Key Findings

1. **Supplier reliability is the #2 driver** of stock-out risk (24% of importance). A supplier arriving 2 days late on a 3-day lead product means zero coverage at the moment of need.
2. **Promotion weeks nearly triple stock-out risk** — promo-week PR-AUC is 0.80 vs 0.51 non-promo.
3. **Beverages and Snacks** are the most predictable high-risk segments. Dairy is the hardest (11% positive rate; model is conservative).
4. **₹4.2 L expected loss** is avoidable for Sep 1–7 if the top 62 HIGH-risk items receive replenishment orders today.
5. **Ponni Rice 5kg at S02** is the single highest-priority item (₹33,168 at risk; order 182 units).

---

## Requirements

```
streamlit>=1.40
plotly>=5.0
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
joblib
```

Install: `pip install streamlit plotly xgboost scikit-learn pandas numpy matplotlib joblib`
