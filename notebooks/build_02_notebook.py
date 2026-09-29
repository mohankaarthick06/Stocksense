"""Builds notebooks/02_features_forecasting.ipynb (run from the repo root, then execute it with nbconvert)."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parent / "02_features_forecasting.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# StockSense - Round 2: Feature Engineering & Demand Forecasting
**Student 2 - ML Engineer** · IntelliData 2026 · NovaMart Retail

Scope from the brief (section 10): *feature engineering, forecasting, validation*.

| Step | What | Code |
|---|---|---|
| 1 | Build one row per forecast origin (close of day *t*, store x product) with 61 leakage-free features and the targets for both models | `src/features.py` |
| 2 | Prove there is no leakage | `features.check_no_leakage()` |
| 3 | Time-aware split with gaps | `src/train_forecast.py` |
| 4 | Baseline, then all permitted regression algorithms, tuned on validation | `train_forecast.tune()` |
| 5 | Test scored once; metrics MAE, RMSE, MAPE, R², WAPE, bias | `train_forecast.evaluate_on_test()` |
| 6 | Residual analysis and permutation importance | |
| 7 | Final model, forecast for 1-7 Sep, out-of-time forecasts for Model 2 | `models/`, `data/processed/` |

The notebook calls the same functions as the scripts, so there is one source of truth. Input: Round 1's `data/processed/` tables.
""")
code("""
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd, numpy as np
from IPython.display import Image, Markdown, display
import features as F
import train_forecast as TF
pd.set_option("display.width", 200, "display.max_columns", 40)
FIG = ROOT / "reports" / "figures"
""")

md("""
## 1. The modelling table: one row per forecast origin

Every evening after closing, for every store x product, the system predicts demand for *t+1 ... t+7*. That moment defines what is "known":
sales, stock and weather up to *t*, purchase orders placed by *t* (with their **expected** date), and calendars published in advance
(planned promotions, holidays, festivals, local events, paydays).

Both targets sit on the same row, so Model 1 and Model 2 describe the same week:
- `next_7_day_demand` = sum of `demand_est` over *t+1..t+7* (Model 1). `demand_est`, not raw sales: on stock-out days sales are capped by the empty shelf.
- `stockout_next7` = 1 if closing stock hits 0 on any day in *t+1..t+7* (Model 2, Student 3).
""")
code("""
m, promos, weather, po = F.load_inputs()
feat = F.build_features(m, promos, weather, po)
print(f"{len(feat):,} origin rows x {len(F.FEATURES)} features")
pd.DataFrame([(g, len(c), ", ".join(c)) for g, c in F.FEATURE_GROUPS.items()],
             columns=["group", "n", "features"])
""")
md("""
Example: Chennai hypermarket, Cola 750 ml, close of 10 June. The item is out of stock that day, 768 units are on order from
SUP04 (late on about a third of its orders), a promotion has just ended (post-promotion dip), and the 1 June MRP hike shows as +9% price change.
""")
code("""
feat[(feat.store_id == "S02") & (feat.product_id == "P330") & (feat.date == "2026-06-10")][
    ["date", "store_id", "product_name"] + F.FEATURES + ["next_7_day_demand", "stockout_next7"]].T
""")

md("""
## 2. Proof of no leakage

The brief: *any feature used to predict the future must contain only information available at prediction time.*
For four origin dates, everything observed after *t* is deleted (sales, stock, weather, PO arrivals), the features are
rebuilt, and each of the 61 features must be identical to the full-data version. Two real leaks were caught this way
while building the table (a festival profile learned over the whole period, and festival dates read from sales data
instead of the published calendar); both were fixed.
""")
code("""
n = F.check_no_leakage(m, promos, weather, po, feat)
print(f"{n:,} origin rows rebuilt from truncated data - all {len(F.FEATURES)} features identical. No leakage.")
""")
code("""
F.main()   # saves data/processed/features.csv and reports/feature_dictionary.md
""")

md("""
## 3. Time-aware split

No random shuffling. Between splits there is a **7-day gap in origins**: a training origin on 12 Jul has a target covering
13-19 Jul, so validation starts on 20 Jul. Without the gap, validation days would sit inside training targets
(leakage through the label).
""")
code("""
pd.DataFrame([(k, a.date(), b.date(), (b + pd.Timedelta(days=7)).date(), len(TF.between(feat, (a, b))))
              for k, (a, b) in TF.SPLITS.items()],
             columns=["split", "first origin", "last origin", "target windows end", "rows"])
""")

md("""
## 4. Train, tune and compare

Baseline first (the brief's *previous 7-day mean*), then every permitted regression algorithm: Linear Regression, Decision Tree,
Random Forest, XGBoost, KNN and SVM (Naive Bayes is a classifier, so it does not apply). Each is tuned on **validation only**.
The best configuration of each is refitted on train + validation and the **test set is scored once**.
The model is chosen on validation WAPE.

This cell runs the full pipeline (about 10 minutes): tuning, test, importance, final refit, out-of-time forecasts, reports.
""")
code("""
out = TF.main()
""")
code("""
tuning = out["tuning"]
tuning.assign(**{c: tuning[c].round(3) for c in ["WAPE", "MAPE", "R2", "Bias"]}).sort_values("WAPE").head(15)
""")
code("""
res = out["test_res"].sort_values("WAPE")
res.assign(**{c: res[c].round(3) for c in ["WAPE", "MAPE", "R2", "Bias"]}, MAE=res.MAE.round(1), RMSE=res.RMSE.round(1))
""")
code("""display(Image(FIG / "11_model1_comparison.png"))""")

md("""
### Which metric matters most to the business?
**WAPE** (sum of absolute errors / sum of demand), with **bias** as the second check. WAPE weights errors by volume, so it
tracks money and replenishment effort. MAPE explodes on slow movers (actual 2, forecast 4 = 100% error). Bias matters because a
systematic under-forecast causes stock-outs even when MAE looks fine. Safety stock absorbs random error, not a systematic shortfall.
""")

md("## 5. Does the forecast track real demand, and where is it weakest?")
code("""display(Image(FIG / "12_actual_vs_forecast.png"))""")
code("""display(Image(FIG / "13_forecast_error_by_segment.png"))""")
code("""display(Image(FIG / "14_residual_analysis.png"))""")

md("""
## 6. What the model relies on (permutation importance, test set)
Each feature group is shuffled across rows and the increase in WAPE is measured. This shows what the model really uses on unseen data.
""")
code("""display(Image(FIG / "15_model1_feature_importance.png"))""")
code("""pd.read_csv(ROOT / "reports" / "model1_feature_importance_groups.csv").round(4)""")

md("""
## 7. Scored against the answer key (evaluation only)
The answer key holds the simulator's true demand. It is git-ignored and was **not** used for training, tuning or selection.
If it is present locally, it checks two claims:
1. training on `demand_est` beats training on raw sales;
2. the forecast holds on the truly unseen week 1-7 Sep.
""")
code("""
p = FIG / "16_answer_key_check.png"
if out["ak"]:
    display(Image(p))
    display(pd.DataFrame(out["ak"]["test_vs_truth"]).T.round(3))
    display(pd.DataFrame(out["ak"]["holdout_week"]).T.round(3))
else:
    print("Answer key not present locally - skipped.")
""")

md("## 8. Final forecast for 1-7 Sep and hand-off")
code("""
fc = out["final"].sort_values("forecast_7d", ascending=False)
fc[["store_id", "product_name", "category", "closing_stock", "incoming_next7", "baseline_7d",
    "forecast_7d", "forecast_p10", "forecast_p90"]].head(12)
""")
code("""
from forecast import load_model, predict
bundle = load_model()
print(bundle["algorithm"], "|", bundle["config"], "| trained on origins", bundle["trained_on_origins"])
today = feat[feat.date == feat.date.max()]
print("re-predicted from the saved model matches the saved forecast:",
      np.allclose(predict(bundle, today), out["final"].forecast_7d, atol=0.051))
""")
md("""
**Hand-off to Student 3 (stock-out risk, explainability, recommendation)**
- `data/processed/features.csv`: same origin rows, with the Model 2 target `stockout_next7` (about 28% positive) and all stock, supply and supplier features.
- `data/processed/forecast_oot.csv`: an **out-of-time** 7-day forecast for every origin from 12 Apr. Each 14-day block was predicted by a model trained only on
  earlier targets, so it can be used as a Model 2 feature (e.g. `coverage = (closing_stock + incoming_next7) / forecast_7d`) without leakage.
- `data/processed/forecast_next7.csv`: the 1-7 Sep forecast with an 80% range; `forecast_sigma_7d` feeds safety stock = z x sigma x sqrt((lead + review) / 7).
- `models/demand_forecast.joblib` + `src/forecast.py`: `load_model()` / `predict()` for the dashboard and what-if analysis.
- Full write-up: `reports/model1_demand_forecast.md`; feature list: `reports/feature_dictionary.md`.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, NB)
print(f"wrote {NB}")
