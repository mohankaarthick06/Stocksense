# StockSense — Round 3: Decision Intelligence

> **IntelliData 2026 — NovaMart Retail**  
> Student 3: Decision Intelligence

---

## Executive Summary

StockSense identifies **62 HIGH-risk** and **122 MEDIUM-risk** SKU-store combinations for the week of Sep 1–7, 2026. If no action is taken, the system estimates **₹420,336** in avoidable lost revenue. The system issues replenishment orders, safety stock targets, and plain-language manager instructions for every item.

---

## Phase 1 — Model 2: Stock-out Risk Classifier

### Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Algorithms | NaiveBayes, Decision Tree, Random Forest, XGBoost, KNN, LinearSVM | Permitted list; Logistic Regression excluded per DECISIONS D2.7 |
| Imbalance handling | class_weight=balanced / scale_pos_weight | 28% positives manageable; SMOTE excluded (time-series structure — synthetic rows would mix market conditions) |
| Calibration | isotonic / sigmoid; pick by Brier score | Class reweighting inflates raw probs; threshold-based rules need reliable probabilities |
| Evaluation metric | PR-AUC (primary), recall@0.40 (tie-break) | Always-no-stock-out = 72% accuracy; PR-AUC floor = base rate |
| OOT stacking | coverage, forecast_vs_rm7, lead_cover_gap from forecast_oot.csv | No leakage: OOT forecast was made before test rows were seen |
| Test protocol | Tuned on val; test set scored once | Standard holdout; no peeking |

### Selected model: LinearSVM(C=1.0)

Calibration: **isotonic**  

| Metric | Test set |
|---|---|
| PR-AUC | **0.6299** |
| ROC-AUC | 0.7988 |
| Recall@0.40 | 0.4989 |
| Precision@0.40 | 0.6281 |
| F1@0.40 | 0.5561 |
| Brier | 0.1372 |

> Rule baseline PR-AUC: 0.3133 — model achieves 2x improvement.

### Algorithm comparison (all candidates, test set)

| Algorithm | PR-AUC | ROC-AUC | Recall@0.40 | F1@0.40 | Brier |
|---|---|---|---|---|---|
| LinearSVM(C=1.0) | 0.6489 | 0.8004 | 0.3738 | 0.4958 | 0.1400 |
| SELECTED+CAL: LinearSVM(C=1.0) | 0.6299 | 0.7988 | 0.4989 | 0.5561 | 0.1372 |
| LinearSVM(C=0.1) | 0.6293 | 0.7877 | 0.3949 | 0.5051 | 0.1413 |
| XGB(n=300,d=4,lr=0.05) | 0.6065 | 0.7800 | 0.5840 | 0.5435 | 0.1508 |
| XGB(n=300,d=4,lr=0.1) | 0.5921 | 0.7775 | 0.5750 | 0.5487 | 0.1514 |
| XGB(n=600,d=4,lr=0.05) | 0.5902 | 0.7669 | 0.5494 | 0.5335 | 0.1525 |
| XGB(n=300,d=6,lr=0.05) | 0.5813 | 0.7674 | 0.5283 | 0.5297 | 0.1524 |
| KNN(k=31) | 0.5802 | 0.7813 | 0.3941 | 0.4849 | 0.1461 |
| RF(n=200,d=10) | 0.5794 | 0.7705 | 0.5592 | 0.5367 | 0.1575 |
| RF(n=400,d=10) | 0.5790 | 0.7722 | 0.5629 | 0.5382 | 0.1571 |
| RF(n=400,d=20) | 0.5764 | 0.7706 | 0.3308 | 0.4414 | 0.1497 |
| RF(n=200,d=20) | 0.5700 | 0.7647 | 0.3338 | 0.4439 | 0.1505 |
| XGB(n=600,d=4,lr=0.1) | 0.5686 | 0.7609 | 0.5343 | 0.5221 | 0.1583 |
| KNN(k=15) | 0.5657 | 0.7725 | 0.5079 | 0.5431 | 0.1467 |
| XGB(n=300,d=6,lr=0.1) | 0.5623 | 0.7512 | 0.4921 | 0.5094 | 0.1583 |
| XGB(n=600,d=6,lr=0.05) | 0.5584 | 0.7471 | 0.4717 | 0.4964 | 0.1593 |
| XGB(n=600,d=6,lr=0.1) | 0.5526 | 0.7419 | 0.4461 | 0.4903 | 0.1650 |
| DecisionTree(d=5,ms=20) | 0.4980 | 0.7049 | 0.5087 | 0.4811 | 0.1701 |
| DecisionTree(d=5,ms=50) | 0.4980 | 0.7049 | 0.5087 | 0.4811 | 0.1701 |
| KNN(k=5) | 0.4788 | 0.7200 | 0.5494 | 0.5109 | 0.1703 |
| NaiveBayes(vs=1e-09) | 0.4258 | 0.7106 | 0.3052 | 0.3672 | 0.2422 |
| DecisionTree(d=10,ms=50) | 0.4163 | 0.6193 | 0.4514 | 0.4439 | 0.2105 |
| DecisionTree(d=10,ms=20) | 0.3902 | 0.6015 | 0.4461 | 0.4343 | 0.2224 |
| DecisionTree(d=None,ms=50) | 0.3337 | 0.6011 | 0.4295 | 0.3781 | 0.2610 |
| NaiveBayes(vs=0.001) | 0.3316 | 0.5991 | 0.2675 | 0.2927 | 0.2111 |
| DecisionTree(d=None,ms=20) | 0.2809 | 0.5564 | 0.3436 | 0.3325 | 0.2972 |
| Rule (stock<reorder) | 0.2697 | 0.5519 | 0.4393 | 0.3562 | 0.3909 |

### Segment analysis (test set)

| Category | PR-AUC | Recall@0.40 | Note |
|---|---|---|---|
| Beverages | 0.84 | 0.82 | Strongest — clear demand spikes (festivals, heat) |
| Snacks | 0.71 | 0.62 | Second best — promo-driven |
| Dairy | 0.25 | 0.05 | Weakest — very low positive rate (11%); model conservative |
| Promo weeks | 0.80 | 0.74 | Promos nearly triple stock-out risk |

---

## Phase 2 — Explainability

### Global: permutation importance

| Feature group | Share of PR-AUC drop |
|---|---|
| Recent sales trend | 33% |
| Incoming stock & supplier reliability | 24% |
| Current stock & cover | 21% |
| Store & product profile | 16% |
| Festival / holiday / payday calendar | 4% |
| Promotion plan | 2% |
| Stock-out history | 0% |
| Price | 0% |

**Key insight:** Incoming stock and supplier reliability jointly account for ~24% of predictive power. A supplier arriving 2 days late on a 3-day lead product means zero stock at the moment of replenishment.

### Local: what-if flip method

For each row, one feature group at a time is reset to a neutral value (no promotion, no festival, supplier always on time). The probability drop is attributed to that driver. Outputs a plain-language sentence per driver.

**Example:** S01 / Cold Coffee Can 200ml (74% risk):
  - Supplier late 35% → +61% risk
  - Promotion 20% off 5 days → +39% risk

---

## Phase 3 — Business Action Layer

All formulas implemented in `src/recommend.py` (pure functions, unit-tested).

| Component | Formula / Rule | Reference |
|---|---|---|
| Adjusted forecast | `forecast_7d − bias_units` (positive bias = over-forecast) | WORKLOG Step 4 / D3.2 |
| Safety stock | `z × σ₇d × √((effective_lead + 1) / 7)` | D3.1 |
| z-score | 1.65 (non-perishable), 1.28 (shelf_life < 7d) | D3.2 |
| Perishable cover | `min(7, shelf_life − 1)` days only | D3.4 |
| Reorder qty | `max(0, recommended_stock − stock − incoming)` | D3.3 |
| Risk level | HIGH ≥ 0.70, MEDIUM ≥ 0.40, LOW < 0.40 | D2.7 |
| Expected loss | `prob × max(0, forecast − stock − incoming) × price` | D3.6 |

### Manager action matrix

| Risk | Reorder > 0 | Action |
|---|---|---|
| HIGH | Yes | Raise replenishment order today |
| HIGH | No | Expedite / confirm open PO |
| MEDIUM | Yes | Add to next order cycle |
| MEDIUM | No | Monitor daily |
| LOW | Yes | Order at normal cycle |
| LOW | No | No action |
| Any | Overstock (DoC > 2×(lead+7)) | Pause orders / consider markdown |

### Sep 1-7 summary

- **62** HIGH-risk items — replenishment needed today
- **122** MEDIUM-risk items — add to next order cycle
- **₹420,336** expected loss if no action taken

---

## Phase 4 — Streamlit Dashboard

Run: `streamlit run dashboard/app.py`

| Page | Content |
|---|---|
| Executive Summary | KPI cards, revenue trend, risk donut, top stores by revenue |
| Demand Intelligence | Actual vs forecast per item, error by category, weekly demand |
| Inventory Risk | Store × category heatmap, risk table, filters |
| Manager Action Centre | Action cards (HIGH expanded), What-If simulator, CSV download |
| Model Performance | Model 1 & 2 comparison tables, all evaluation charts |
| Explainability | Group permutation importance, per-item driver sentences |

**What-If simulator:** adjust discount %, supplier delay days, and festival multiplier for any item. See probability, risk level, reorder qty, and expected loss update live.

---

## Generated Outputs

### Models

| File | Description |
|---|---|
| `models/stockout_model.joblib` | Calibrated stock-out classifier bundle (model + levels + thresholds) |
| `models/stockout_model_meta.json` | Algorithm name, test metrics, feature list, thresholds |

### Reports

| File | Description |
|---|---|
| `reports/model2_tuning_log.csv` | All candidate models scored on validation set |
| `reports/model2_comparison.csv` | All candidates scored on test set |
| `reports/model2_feature_importance.csv` | Feature-level permutation importance |
| `reports/model2_feature_importance_groups.csv` | Group-level permutation importance |
| `reports/model2_stockout_risk.md` | Narrative report: design, comparison, segments, limitations |
| `data/processed/recommendations_1_7_sep.csv` | 387-row Sep 1-7 action table |

### Figures

| Figure | Description |
|---|---|
| Fig 17 | Calibration curve: uncalibrated vs calibrated |
| Fig 18 | ROC and PR curves (test set, all algorithms) |
| Fig 19 | Confusion matrices at threshold 0.40 and 0.70 |
| Fig 20 | Algorithm comparison bar chart (test set) |
| Fig 21 | Feature group permutation importance |

---

## Source Files (Student 3)

| File | Purpose |
|---|---|
| `src/train_stockout.py` | Phase 1: Train, tune, calibrate, evaluate, save Model 2 |
| `src/stockout.py` | Load/predict helpers for Model 2 |
| `src/explain.py` | Global permutation importance + local what-if flip |
| `src/recommend.py` | Business action layer (pure functions, unit tested) |
| `src/make_recommendations.py` | Build Sep 1-7 action table |
| `dashboard/app.py` | Streamlit dashboard (6 pages) |
| `dashboard/data_access.py` | Cached data loaders |
| `dashboard/components.py` | Reusable Plotly charts and UI components |
| `tests/test_recommend.py` | 9 unit tests for business action layer |
| `scripts/generate_reports.py` | This report generator |
