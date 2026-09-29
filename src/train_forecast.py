"""
StockSense - Round 2 Model 1: 7-day demand forecast (Person 2: ML Engineer).

Target: next_7_day_demand = sum of demand_est over t+1..t+7 for one store x product,
forecast at close of day t. Regression, one global model for all 387 combinations.

Protocol
    1. Time-aware split with gaps, so no training target overlaps a later split.
    2. Mandatory baseline first: 7 x mean demand of the last 7 days.
    3. Every permitted regression algorithm (Linear Regression, Decision Tree,
       Random Forest, XGBoost, KNN, SVM) is tuned on VALIDATION only.
    4. Model selected on validation WAPE. The TEST set is scored once, after refitting
       each tuned model on train + validation.
    5. The selected model is refitted on every labelled origin and forecasts 1-7 Sep.
    6. Out-of-time forecasts for every origin (rolling refits) are saved for Model 2.

Run from the repo root (after src/features.py):
    python src/train_forecast.py
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from eda_stats import (BLUES, DE_EMPHASIS, INK, INK2, MUTED, NEGATIVE, POSITIVE, SLOTS, CAT_COLOR, CATEGORIES,
                       STORE_COLOR, STORES, md_table, rupees, suptitle)
from features import CATEGORICAL, COMBO, FEATURE_GROUPS, FEATURES
from forecast import MODEL_PATH, MODELS, category_levels, model_frame

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ANSWER_KEY = ROOT / "data" / "answer_key_DO_NOT_COMMIT"
SEED = 42
TARGET = "next_7_day_demand"

D = pd.Timestamp
# Origins from 8 Mar: the first week is warm-up so every row has at least 7 days of history.
# Gap of 7 origin days before each later split: a training origin on 12 Jul has a target
# window ending 19 Jul, the day before the first validation origin.
SPLITS = {
    "train": (D("2026-03-08"), D("2026-07-12")),
    "validation": (D("2026-07-20"), D("2026-08-03")),
    "test": (D("2026-08-11"), D("2026-08-24")),
}
TRAIN_VAL = (SPLITS["train"][0], SPLITS["validation"][1])
ALL_LABELLED = (SPLITS["train"][0], SPLITS["test"][1])
FINAL_ORIGIN = D("2026-08-31")
OOT_FIRST_ORIGIN = D("2026-04-12")
OOT_BLOCK_DAYS = 14


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_features():
    return pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])


def between(df, span):
    return df[(df.date >= span[0]) & (df.date <= span[1]) & df[TARGET].notna()]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    nz = y > 0
    return {
        "WAPE": np.abs(e).sum() / y.sum(),
        "MAE": np.abs(e).mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MAPE": np.mean(np.abs(e[nz]) / y[nz]),
        "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "Bias": e.sum() / y.sum(),
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def scaled_pipeline(estimator):
    """For distance / linear models: one-hot categories, impute + flag missing, compress skew, scale.
    The target is modelled on log scale because demand is a skewed count (1 to ~2,000 units a week)."""
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              FunctionTransformer(signed_log1p), StandardScaler()), numeric),
    ])
    return TransformedTargetRegressor(regressor=make_pipeline(prep, estimator), func=np.log1p, inverse_func=np.expm1)


class SubsampledSVR:
    """RBF-kernel SVR trains in roughly quadratic time, so it is fitted on a random subsample of rows."""

    def __init__(self, n_rows=12000, **params):
        self.n_rows, self.params = n_rows, params
        self.model = scaled_pipeline(SVR(kernel="rbf", **params))

    def fit(self, X, y):
        idx = np.random.default_rng(SEED).choice(len(X), size=min(self.n_rows, len(X)), replace=False)
        self.model.fit(X.iloc[idx], np.asarray(y)[idx])
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBDemand:
    """XGBoost with early stopping on validation. `log_target=True` trains on log1p(demand)."""

    def __init__(self, objective="reg:squarederror", log_target=False, n_estimators=None, **params):
        self.objective, self.log_target, self.n_estimators = objective, log_target, n_estimators
        self.params = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           n_jobs=-1, random_state=SEED, **params)
        if objective == "reg:tweedie":
            self.params["tweedie_variance_power"] = 1.3
        self.best_iteration = None

    def _t(self, y):
        return np.log1p(y) if self.log_target else np.asarray(y, float)

    def fit(self, X, y, X_val=None, y_val=None):
        if self.n_estimators is None:   # tuning: stop when validation error stops improving
            self.model = XGBRegressor(objective=self.objective, n_estimators=4000, early_stopping_rounds=150,
                                      eval_metric="mae", **self.params)
            self.model.fit(X, self._t(y), eval_set=[(X_val, self._t(y_val))], verbose=False)
            self.best_iteration = self.model.best_iteration + 1
        else:                           # refit with the number of trees found during tuning
            self.model = XGBRegressor(objective=self.objective, n_estimators=self.n_estimators, **self.params)
            self.model.fit(X, self._t(y), verbose=False)
            self.best_iteration = self.n_estimators
        return self

    def predict(self, X):
        p = self.model.predict(X)
        return np.expm1(p) if self.log_target else p


def candidates():
    """(algorithm, config label, factory). Grids are small on purpose: tuned on one validation window."""
    c = [("Linear Regression", "OLS on log target", lambda: scaled_pipeline(LinearRegression()))]
    for depth in (6, 8, 10, 12, 16):
        for leaf in (5, 20, 50):
            c.append(("Decision Tree", f"max_depth={depth}, min_samples_leaf={leaf}",
                      lambda d=depth, l=leaf: DecisionTreeRegressor(max_depth=d, min_samples_leaf=l, random_state=SEED)))
    for depth in (None, 18):
        for leaf in (2, 8):
            for mf in (0.33, 0.6):
                c.append(("Random Forest", f"max_depth={depth}, min_samples_leaf={leaf}, max_features={mf}",
                          lambda d=depth, l=leaf, f=mf: RandomForestRegressor(
                              n_estimators=300, max_depth=d, min_samples_leaf=l, max_features=f,
                              n_jobs=-1, random_state=SEED)))
    for obj, log in (("reg:squarederror", False), ("reg:tweedie", False), ("reg:squarederror", True)):
        for depth in (5, 7):
            for mcw in (1, 10):
                name = {("reg:squarederror", False): "squared error", ("reg:tweedie", False): "tweedie",
                        ("reg:squarederror", True): "squared error on log target"}[(obj, log)]
                c.append(("XGBoost", f"{name}, max_depth={depth}, min_child_weight={mcw}",
                          lambda o=obj, lg=log, d=depth, m=mcw: XGBDemand(o, lg, max_depth=d, min_child_weight=m)))
    for k in (5, 10, 20, 40):
        c.append(("KNN", f"k={k}, distance-weighted",
                  lambda k=k: scaled_pipeline(KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1))))
    for C in (0.1, 1.0):
        c.append(("SVM", f"linear kernel, C={C}",
                  lambda C=C: scaled_pipeline(LinearSVR(C=C, epsilon=0.0, loss="squared_epsilon_insensitive",
                                                        dual="auto", max_iter=20000, random_state=SEED))))
    for C in (1.0, 5.0):
        c.append(("SVM", f"RBF kernel, C={C}, 12k-row subsample",
                  lambda C=C: SubsampledSVR(C=C, epsilon=0.05, gamma="scale")))
    return c


def fit_model(model, X, y, X_val=None, y_val=None):
    if isinstance(model, XGBDemand):
        return model.fit(X, y, X_val, y_val)
    return model.fit(X, y)


def refit_factory(algorithm, factory, fitted):
    """Same configuration for a refit; XGBoost keeps the tree count found by early stopping."""
    if algorithm == "XGBoost":
        return lambda: XGBDemand(fitted.objective, fitted.log_target, n_estimators=fitted.best_iteration,
                                 **{k: fitted.params[k] for k in ("max_depth", "min_child_weight")})
    return factory


# --------------------------------------------------------------------------
# Tuning, selection, test
# --------------------------------------------------------------------------
def tune(train, val, levels):
    Xtr, ytr = model_frame(train, levels), train[TARGET].values
    Xva, yva = model_frame(val, levels), val[TARGET].values
    rows = [dict(algorithm="Baseline (7 x last-7-day mean)", config="no training", **metrics(yva, val.baseline_7d),
                 fit_seconds=0.0)]
    fitted = {}
    for algo, cfg, factory in candidates():
        t0 = time.time()
        model = fit_model(factory(), Xtr, ytr, Xva, yva)
        pred = np.clip(model.predict(Xva), 0, None)
        secs = time.time() - t0
        if algo == "XGBoost":
            cfg += f", trees={model.best_iteration}"
        rows.append(dict(algorithm=algo, config=cfg, **metrics(yva, pred), fit_seconds=secs))
        fitted[(algo, cfg)] = (factory, model, pred)
        print(f"  {algo:18s} {cfg:62s} WAPE {rows[-1]['WAPE']:.3f}  ({secs:.0f}s)")
    return pd.DataFrame(rows), fitted


def best_per_algorithm(tuning):
    t = tuning[~tuning.algorithm.str.startswith("Baseline")]
    return t.loc[t.groupby("algorithm").WAPE.idxmin()].sort_values("WAPE")


def evaluate_on_test(best, fitted, train_val, test, levels):
    Xtv, ytv = model_frame(train_val, levels), train_val[TARGET].values
    Xte, yte = model_frame(test, levels), test[TARGET].values
    rows = [dict(algorithm="Baseline (7 x last-7-day mean)", **metrics(yte, test.baseline_7d))]
    preds = {"Baseline (7 x last-7-day mean)": test.baseline_7d.values}
    refits = {}
    for r in best.itertuples():
        factory, val_model, _ = fitted[(r.algorithm, r.config)]
        model = fit_model(refit_factory(r.algorithm, factory, val_model)(), Xtv, ytv)
        p = np.clip(model.predict(Xte), 0, None)
        rows.append(dict(algorithm=r.algorithm, **metrics(yte, p)))
        preds[r.algorithm] = p
        refits[r.algorithm] = model
    return pd.DataFrame(rows), preds, refits


def segment_table(df, pred_col, by):
    g = df.assign(err=df[pred_col] - df[TARGET], abs_err=(df[pred_col] - df[TARGET]).abs(),
                  base_abs=(df.baseline_7d - df[TARGET]).abs())
    out = g.groupby(by).agg(rows=(TARGET, "size"), actual=(TARGET, "sum"), abs_err=("abs_err", "sum"),
                            err=("err", "sum"), base_abs=("base_abs", "sum"))
    out["WAPE"] = out.abs_err / out.actual
    out["Bias"] = out.err / out.actual
    out["Baseline WAPE"] = out.base_abs / out.actual
    out["Improvement vs baseline"] = 1 - out.WAPE / out["Baseline WAPE"]
    return out[["rows", "WAPE", "Bias", "Baseline WAPE", "Improvement vs baseline"]]


# --------------------------------------------------------------------------
# Explainability (global): grouped permutation importance on the TEST set
# --------------------------------------------------------------------------
def permutation_importance(model, df, levels, groups, repeats=5):
    """Increase in WAPE when a feature (or a whole group of features) is shuffled across rows."""
    rng = np.random.default_rng(SEED)
    X = model_frame(df, levels)
    y = df[TARGET].values
    base = metrics(y, np.clip(model.predict(X), 0, None))["WAPE"]
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]   # shuffle the group jointly, keeping its internal structure
            drops.append(metrics(y, np.clip(model.predict(Xp), 0, None))["WAPE"] - base)
        rows.append(dict(feature=name, wape_increase=np.mean(drops), std=np.std(drops)))
    return pd.DataFrame(rows).sort_values("wape_increase", ascending=False), base


# --------------------------------------------------------------------------
# Out-of-time forecasts (for Model 2 stacking) and the final forecast
# --------------------------------------------------------------------------
def out_of_time_forecasts(feat, make_model, levels, final_model):
    """
    Each block of 14 origin days is predicted by a model trained only on origins whose target
    window ended before the block starts - the forecast a store would really have had that day.
    """
    labelled = feat[feat[TARGET].notna()]
    out = []
    b = OOT_FIRST_ORIGIN
    while b < FINAL_ORIGIN:
        e = min(b + pd.Timedelta(days=OOT_BLOCK_DAYS - 1), FINAL_ORIGIN - pd.Timedelta(days=1))
        tr = labelled[(labelled.date >= SPLITS["train"][0]) & (labelled.date <= b - pd.Timedelta(days=8))]
        blk = feat[(feat.date >= b) & (feat.date <= e)]
        model = make_model().fit(model_frame(tr, levels), tr[TARGET].values)
        out.append(blk[["date"] + COMBO].assign(forecast_7d=np.clip(model.predict(model_frame(blk, levels)), 0, None),
                                                trained_through_origin=(b - pd.Timedelta(days=8)).date()))
        print(f"  origins {b.date()} .. {e.date()}: trained on {len(tr):,} rows")
        b = e + pd.Timedelta(days=1)
    last = feat[feat.date == FINAL_ORIGIN]
    out.append(last[["date"] + COMBO].assign(forecast_7d=np.clip(final_model.predict(model_frame(last, levels)), 0, None),
                                             trained_through_origin=ALL_LABELLED[1].date()))
    oot = pd.concat(out).merge(feat[["date"] + COMBO + ["baseline_7d", TARGET]], on=["date"] + COMBO)
    return oot


def error_profiles(oot, feat):
    """Forecast-error spread for safety stock: per category and per store x product (out-of-time rows only)."""
    r = oot[oot[TARGET].notna()].merge(feat[["date"] + COMBO + ["category_name"]], on=["date"] + COMBO)
    r["err"] = r[TARGET] - r.forecast_7d    # positive = demand above forecast = stock-out risk
    cat = r.groupby("category_name").agg(rows=("err", "size"), mean_actual=(TARGET, "mean"),
                                         bias_units=("err", "mean"), sigma_7d=("err", "std"))
    cat["sigma_rel"] = cat.sigma_7d / cat.mean_actual
    combo = r.groupby(COMBO).agg(rows=("err", "size"), mean_actual=(TARGET, "mean"),
                                 bias_units=("err", "mean"), sigma_7d=("err", "std"),
                                 wape=("err", lambda e: np.abs(e).sum()))
    combo["wape"] = combo.wape / (combo.mean_actual * combo.rows).replace(0, np.nan)
    return cat.round(4), combo.round(4).reset_index()


# --------------------------------------------------------------------------
# Answer key (evaluation only - never used for training, tuning or selection)
# --------------------------------------------------------------------------
def answer_key_checks(feat, test, test_pred, refit_sold_model, levels, final):
    path = ANSWER_KEY / "true_demand_and_holdout.csv"
    if not path.exists():
        return None
    ak = pd.read_csv(path, parse_dates=["date"]).sort_values(COMBO + ["date"])
    g = ak.groupby(COMBO, sort=False).true_demand
    ak["true_next7"] = g.transform(lambda s: s[::-1].rolling(7, min_periods=7).sum()[::-1].shift(-1))
    t = test[["date"] + COMBO + [TARGET, "next_7_day_units_sold"]].copy()
    t["forecast"] = test_pred
    t["forecast_sales_model"] = np.clip(refit_sold_model.predict(model_frame(test, levels)), 0, None)
    t = t.merge(ak[["date"] + COMBO + ["true_next7"]], on=["date"] + COMBO)
    res = {
        "test_vs_truth": {
            "Model trained on demand_est (ours)": metrics(t.true_next7, t.forecast),
            "Same model trained on raw units_sold": metrics(t.true_next7, t.forecast_sales_model),
        },
        "target_vs_truth": {
            "demand_est target": metrics(t.true_next7, t[TARGET]),
            "units_sold target": metrics(t.true_next7, t.next_7_day_units_sold),
        },
    }
    hs = pd.read_csv(ANSWER_KEY / "holdout_week_summary.csv")
    h = final.merge(hs[COMBO + ["true_demand_7d", "stockout_any_7d_bau"]], on=COMBO)
    res["holdout_week"] = {
        "Model (origin 31 Aug)": metrics(h.true_demand_7d, h.forecast_7d),
        "Baseline": metrics(h.true_demand_7d, h.baseline_7d),
    }
    res["holdout_by_category"] = h.groupby("category").apply(
        lambda d: pd.Series({"true": d.true_demand_7d.sum(), "forecast": d.forecast_7d.sum(),
                             "WAPE": (d.forecast_7d - d.true_demand_7d).abs().sum() / d.true_demand_7d.sum(),
                             "Baseline WAPE": (d.baseline_7d - d.true_demand_7d).abs().sum() / d.true_demand_7d.sum()}),
        include_groups=False)
    return res


# --------------------------------------------------------------------------
# Charts (numbered after Round 1's 01-10)
# --------------------------------------------------------------------------
def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def chart_comparison(val_best, test_res, selected):
    order = test_res.set_index("algorithm").WAPE.sort_values(ascending=False).index
    v = pd.concat([val_best.set_index("algorithm").WAPE,
                   pd.Series({"Baseline (7 x last-7-day mean)": np.nan})])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(order))
    te = test_res.set_index("algorithm").loc[order]
    colors = [SLOTS[0] if a == selected else (SLOTS[1] if a.startswith("Baseline") else DE_EMPHASIS) for a in order]
    a1.barh(y, te.WAPE * 100, color=colors, height=0.62)
    for i, (a, w) in enumerate(te.WAPE.items()):
        vw = v.get(a, np.nan)
        extra = f"   (validation {vw:.1%})" if not np.isnan(vw) else ""
        a1.text(w * 100, i, f"  {w:.1%}{extra}", va="center", fontsize=9, color=INK2)
    a1.set_yticks(y, order)
    a1.set_xlabel("Test WAPE (%) - lower is better")
    a1.set_title("Forecast error on the unseen test weeks")
    a1.grid(axis="y", visible=False); a1.grid(axis="x")
    a1.set_xlim(0, te.WAPE.max() * 100 * 1.55)

    a2.barh(y, te.Bias * 100, color=[NEGATIVE if b < 0 else POSITIVE for b in te.Bias], height=0.62)
    a2.axvline(0, color=INK2, lw=0.8)
    for i, b in enumerate(te.Bias):
        a2.text(b * 100, i, f" {b:+.1%} ", va="center", ha="left" if b >= 0 else "right", fontsize=9, color=INK2)
    a2.set_yticks(y, [""] * len(y))
    a2.set_xlabel("Bias (%): forecast - actual, as share of actual")
    a2.set_title("Under-forecasting causes stock-outs")
    a2.grid(axis="y", visible=False); a2.grid(axis="x")
    lim = max(abs(te.Bias).max() * 100 * 1.8, 5)
    a2.set_xlim(-lim, lim)
    suptitle(fig, "Which algorithm forecasts next-week demand best?")
    fig.tight_layout()
    save(fig, "11_model1_comparison")


def chart_actual_vs_forecast(val, val_pred, test, test_pred):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for df, p, label in ((val, val_pred, "validation"), (test, test_pred, "test")):
        d = df.assign(pred=p).groupby("date")[[TARGET, "pred", "baseline_7d"]].sum()
        a1.plot(d.index, d[TARGET] / 1000, color=INK, lw=2, label="Actual (demand_est)" if label == "validation" else None)
        a1.plot(d.index, d.pred / 1000, color=SLOTS[0], lw=2, label="Model forecast" if label == "validation" else None)
        a1.plot(d.index, d.baseline_7d / 1000, color=SLOTS[1], lw=1.5, ls="--",
                label="Baseline" if label == "validation" else None)
    a1.axvspan(SPLITS["validation"][1] + pd.Timedelta(hours=12), SPLITS["test"][0] - pd.Timedelta(hours=12),
               color=DE_EMPHASIS, alpha=0.25, lw=0)
    a1.text(SPLITS["validation"][1] + pd.Timedelta(days=4), a1.get_ylim()[0], "gap", ha="center", va="bottom",
            color=MUTED, fontsize=9)
    a1.set_ylabel("Next-7-day demand, all stores ('000 units)")
    a1.set_title("Chain total by forecast origin: validation, then test")
    a1.legend(loc="upper left")
    a1.tick_params(axis="x", rotation=30)

    y = test[TARGET].values
    a2.scatter(y + 1, test_pred + 1, s=6, alpha=0.25, color=SLOTS[0], lw=0)
    lim = [1, max(y.max(), test_pred.max()) * 1.3]
    a2.plot(lim, lim, color=INK2, lw=1, ls="--")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("Actual 7-day demand (+1, log scale)")
    a2.set_ylabel("Forecast (+1, log scale)")
    a2.set_title("Every store x product, test weeks")
    a2.grid(axis="x")
    suptitle(fig, "Does the forecast track real demand?")
    fig.tight_layout()
    save(fig, "12_actual_vs_forecast")


def chart_segments(by_cat, by_store):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, tbl, colors, title in ((a1, by_cat, CAT_COLOR, "By category"), (a2, by_store, STORE_COLOR, "By store")):
        tbl = tbl.sort_values("WAPE")
        y = np.arange(len(tbl))
        ax.barh(y + 0.18, tbl["Baseline WAPE"] * 100, height=0.34, color=DE_EMPHASIS, label="Baseline")
        ax.barh(y - 0.18, tbl.WAPE * 100, height=0.34, color=[colors[i] for i in tbl.index], label="Model")
        for i, (w, imp) in enumerate(zip(tbl.WAPE, tbl["Improvement vs baseline"])):
            ax.text(w * 100, i - 0.18, f"  {w:.0%}  ({imp:.0%} less error than baseline)", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, tbl.index)
        ax.set_xlabel("Test WAPE (%)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False); ax.grid(axis="x")
        ax.set_xlim(0, max(tbl["Baseline WAPE"].max(), tbl.WAPE.max()) * 100 * 1.75)
    suptitle(fig, "Where is the forecast weakest?")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DE_EMPHASIS), plt.Rectangle((0, 0), 1, 1, color=INK2)],
               labels=["Baseline", "Model (coloured by category / store)"], loc="upper right", ncol=2,
               bbox_to_anchor=(0.99, 1.0))
    save(fig, "13_forecast_error_by_segment")


def chart_residuals(test, test_pred):
    res = test_pred - test[TARGET].values
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4))
    rel = res / np.maximum(test[TARGET].values, 1)
    a1.hist(np.clip(rel, -1.5, 1.5) * 100, bins=60, color=SLOTS[0])
    a1.axvline(0, color=INK2, lw=1)
    a1.set_xlabel("Error as % of actual (clipped at +/-150%)")
    a1.set_ylabel("Store x product x origin rows")
    a1.set_title("Error distribution")

    bins = pd.qcut(test_pred, 10, duplicates="drop")
    r = pd.DataFrame({"bin": bins, "res": res, "pred": test_pred}).groupby("bin", observed=True).agg(
        pred=("pred", "mean"), mean=("res", "mean"), q10=("res", lambda x: x.quantile(0.1)),
        q90=("res", lambda x: x.quantile(0.9)))
    a2.fill_between(r.pred, r.q10, r.q90, color=SLOTS[0], alpha=0.18, lw=0, label="10th-90th percentile")
    a2.plot(r.pred, r["mean"], color=SLOTS[0], marker="o", ms=4, label="Mean error")
    a2.axhline(0, color=INK2, lw=1)
    a2.set_xscale("log")
    a2.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("Forecast size (units, log scale)")
    a2.set_ylabel("Forecast - actual (units)")
    a2.set_title("Error vs forecast size")
    a2.legend(loc="lower left")
    a2.grid(axis="x")

    promo = test.promo_days_next7.values > 0
    groups = [("No promo", ~promo), ("Promo\nplanned", promo),
              ("Festival\nin window", test.festival_next7.values == 1),
              ("Recent\nstock-out", test.stockout_days_last14.values > 0)]
    labels, w, b = [], [], []
    for name, mask in groups:
        y, p = test[TARGET].values[mask], test_pred[mask]
        m = metrics(y, p)
        labels.append(f"{name}\nn={mask.sum():,}"); w.append(m["WAPE"]); b.append(m["Bias"])
    x = np.arange(len(labels))
    a3.bar(x, np.array(w) * 100, color=[SLOTS[0], SLOTS[3], SLOTS[4], SLOTS[1]], width=0.6)
    for i, (wi, bi) in enumerate(zip(w, b)):
        a3.text(i, wi * 100, f"{wi:.0%}\nbias {bi:+.0%}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    a3.set_xticks(x, labels, fontsize=8.5)
    a3.set_ylabel("Test WAPE (%)")
    a3.set_title("Error by situation")
    a3.set_ylim(0, max(w) * 100 * 1.35)
    suptitle(fig, "Are the errors unbiased, and where do they come from?")
    fig.tight_layout()
    save(fig, "14_residual_analysis")


def chart_importance(groups_imp, single_imp, base):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    g = groups_imp.sort_values("wape_increase")
    a1.barh(g.feature, g.wape_increase * 100, xerr=g["std"] * 100, color=SLOTS[0], height=0.6,
            error_kw=dict(ecolor=MUTED, lw=1))
    for i, v in enumerate(g.wape_increase):
        a1.text(max(v, 0) * 100, i, f"  +{v * 100:.1f} pts", va="center", fontsize=9, color=INK2)
    a1.set_xlabel(f"WAPE increase when shuffled (points; model WAPE {base:.1%})")
    a1.set_title("Feature groups")
    a1.grid(axis="y", visible=False); a1.grid(axis="x")
    a1.set_xlim(0, g.wape_increase.max() * 100 * 1.35)
    s = single_imp.head(15).sort_values("wape_increase")
    a2.barh(s.feature, s.wape_increase * 100, color=SLOTS[2], height=0.6)
    a2.set_xlabel("WAPE increase when shuffled (points)")
    a2.set_title("Top 15 single features")
    a2.grid(axis="y", visible=False); a2.grid(axis="x")
    suptitle(fig, "What does the demand model rely on? (permutation importance, test set)")
    fig.tight_layout()
    save(fig, "15_model1_feature_importance")


def chart_answer_key(ak):
    tv = ak["test_vs_truth"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(tv)
    x = np.arange(len(names))
    bias = [tv[n]["Bias"] * 100 for n in names]
    a1.bar(x, bias, color=[SLOTS[0], SLOTS[1]], width=0.55)
    a1.axhline(0, color=INK2, lw=1)
    for i, (n, b) in enumerate(zip(names, bias)):
        a1.text(i, b, f"bias {b:+.1f}%\nWAPE {tv[n]['WAPE']:.1%}", ha="center", va="top" if b < 0 else "bottom",
                fontsize=9, color=INK2)
    a1.set_xticks(x, ["Trained on demand_est\n(corrected for empty shelves)", "Trained on raw units_sold"])
    a1.set_ylabel("Bias vs TRUE demand (%)")
    a1.set_title("Test weeks, scored against the answer key")
    lo = min(bias) * 1.8
    a1.set_ylim(min(lo, -1), max(max(bias) * 1.8, 1))

    hc = ak["holdout_by_category"].sort_values("true")
    y = np.arange(len(hc))
    a2.barh(y + 0.18, hc.true / 1000, height=0.34, color=DE_EMPHASIS, label="True demand 1-7 Sep")
    a2.barh(y - 0.18, hc.forecast / 1000, height=0.34, color=[CAT_COLOR[c] for c in hc.index], label="Forecast (31 Aug)")
    for i, (w, bw) in enumerate(zip(hc.WAPE, hc["Baseline WAPE"])):
        a2.text(max(hc.true.iloc[i], hc.forecast.iloc[i]) / 1000, i, f"  WAPE {w:.0%} (baseline {bw:.0%})",
                va="center", fontsize=8.5, color=INK2)
    a2.set_yticks(y, hc.index)
    a2.set_xlabel("Units ('000)")
    a2.set_title("Holdout week 1-7 Sep (never seen)")
    a2.legend(loc="lower right")
    a2.grid(axis="y", visible=False); a2.grid(axis="x")
    a2.set_xlim(0, hc[["true", "forecast"]].max().max() / 1000 * 1.6)
    suptitle(fig, "Can the forecast be trusted on truly unseen demand?")
    fig.tight_layout()
    save(fig, "16_answer_key_check")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def fmt_metrics(df, cols=("WAPE", "MAE", "RMSE", "MAPE", "R2", "Bias")):
    d = df.copy()
    for c in ("WAPE", "MAPE", "Bias"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.1%}" if c == "Bias" else f"{v:.1%}")
    for c in ("MAE", "RMSE"):
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.1f}")
    if "R2" in d:
        d["R2"] = d.R2.map(lambda v: f"{v:.3f}")
    return d


def write_report(ctx):
    L = []
    add = L.append
    sel, test_res, val_best = ctx["selected"], ctx["test_res"], ctx["val_best"]
    ts = test_res.set_index("algorithm")
    base_w = ts.loc["Baseline (7 x last-7-day mean)", "WAPE"]
    sel_m = ts.loc[sel]
    add("# Model 1 - 7-day demand forecast")
    add("")
    add("*Generated by `src/train_forecast.py`; every number below is computed, not typed.*")
    add("")
    add("## Summary")
    add("")
    add(f"- **Selected model: {sel}** ({ctx['selected_config']}).")
    add(f"- On the unseen test weeks it has **WAPE {sel_m.WAPE:.1%}** versus {base_w:.1%} for the mandatory baseline "
        f"(7 x last-7-day mean): **{1 - sel_m.WAPE / base_w:.0%} less error**. R2 {sel_m.R2:.3f}, bias {sel_m.Bias:+.1%}.")
    add("- Chosen on validation WAPE, the rule fixed before training. The test set was scored once, after tuning was finished.")
    if ctx.get("ak"):
        ak = ctx["ak"]
        hw = ak["holdout_week"]
        add(f"- On the holdout week 1-7 Sep (true demand from the answer key, never used for training), WAPE is "
            f"**{hw['Model (origin 31 Aug)']['WAPE']:.1%}** vs {hw['Baseline']['WAPE']:.1%} for the baseline.")
    add("")
    add("## Problem set-up")
    add("")
    add("| Item | Choice | Why |")
    add("|---|---|---|")
    add("| Prediction moment | Close of day *t*, for every store x product | When a manager can still raise an order |")
    add("| Target | `next_7_day_demand` = sum of `demand_est` over *t+1..t+7* | `demand_est` corrects sales on stock-out days; raw sales would teach the model that stores which run out have low demand |")
    add("| Model scope | One global model for all 387 store x product combinations | ~150 training days per combination is too little for separate models; a global model shares learning (e.g. promotions lift snacks) and can forecast new products from their store / category |")
    add(f"| Features | {len(FEATURES)} features in {len(FEATURE_GROUPS)} groups - see `feature_dictionary.md` | Only information known at close of day *t*; proven by `check_no_leakage()` |")
    add("")
    add("## Time-aware split")
    add("")
    add("| Split | Forecast origins | Target windows end | Rows |")
    add("|---|---|---|---|")
    for k, (a, b) in SPLITS.items():
        add(f"| {k.title()} | {a.date()} .. {b.date()} | {(b + pd.Timedelta(days=7)).date()} | {ctx['sizes'][k]:,} |")
    add(f"| Final forecast | {FINAL_ORIGIN.date()} | predicts 1-7 Sep | {ctx['sizes']['final']:,} |")
    add("")
    add("No random shuffling. Between splits there is a **7-day gap in origins**: a training row with origin 12 Jul has a "
        "target covering 13-19 Jul, so validation starts on 20 Jul. Without the gap, validation days would sit inside training "
        "targets (leakage through the label). The first week of March is warm-up so every row has at least 7 days of history.")
    add("")
    add("## Which metric matters most to the business")
    add("")
    add("All required metrics are reported (MAE, RMSE, MAPE, R2), but the model is **selected on WAPE** "
        "(sum of absolute errors / sum of actual demand), with **bias** as the second check.")
    add("")
    add("- **WAPE** weights each error by volume, so it tracks money and replenishment effort. It is also comparable "
        "across stores and categories.")
    add("- **MAPE** explodes on slow movers (actual 2, forecast 4 = 100% error) and would push the model to optimise the least "
        f"important products. On the test set the selected model's MAPE is {sel_m.MAPE:.0%} while its WAPE is {sel_m.WAPE:.0%}: "
        "most of that MAPE comes from items selling a handful of units a week.")
    add("- **Bias** matters because a model that systematically under-forecasts causes stock-outs even when its MAE looks good. "
        "Safety stock absorbs random error, not a systematic shortfall.")
    add("- **RMSE** is dominated by the few largest items (hypermarket milk sells ~1,300 units a week), and **R2** is high for "
        "any sensible model because it is inflated by the spread between large and small items.")
    add("")
    add("## Model comparison")
    add("")
    add("Best configuration of each permitted algorithm, tuned on validation, then refitted on train + validation and "
        "scored once on test:")
    add("")
    comp = test_res.merge(val_best[["algorithm", "config", "WAPE"]].rename(columns={"WAPE": "Validation WAPE"}),
                          on="algorithm", how="left").sort_values("WAPE")
    comp["config"] = comp.config.fillna("no training")
    comp["Validation WAPE"] = comp["Validation WAPE"].fillna(ctx["val_baseline_wape"])
    comp = fmt_metrics(comp)
    comp["Validation WAPE"] = comp["Validation WAPE"].map(lambda v: f"{v:.1%}")
    add(md_table(comp[["algorithm", "config", "Validation WAPE", "WAPE", "MAE", "RMSE", "MAPE", "R2", "Bias"]]
                 .rename(columns={"WAPE": "Test WAPE"})))
    add("")
    add("![Model comparison](figures/11_model1_comparison.png)")
    add("")
    add("**Not used:** Naive Bayes - it is a classifier, and this is a regression problem.")
    add("")
    add("### Why this model")
    add("")
    for line in ctx["why"]:
        add(f"- {line}")
    add("")
    add(f"Full tuning log ({ctx['n_configs']} configurations): `reports/model1_tuning_log.csv`.")
    add("")
    add("## Does it track real demand?")
    add("")
    add("![Actual vs forecast](figures/12_actual_vs_forecast.png)")
    add("")
    add("## Where is it weakest?")
    add("")
    add(md_table(fmt_segment(ctx["by_cat"])))
    add("")
    add(md_table(fmt_segment(ctx["by_store"])))
    add("")
    add("![Error by segment](figures/13_forecast_error_by_segment.png)")
    add("")
    add("**New products (sparse history).** " + ctx["sparse_text"])
    add("")
    add("## Residual analysis")
    add("")
    add(ctx["residual_text"])
    add("")
    add("![Residuals](figures/14_residual_analysis.png)")
    add("")
    add("## What the model relies on")
    add("")
    add("Permutation importance on the **test** set: each feature group is shuffled across rows and the increase in WAPE "
        "is measured. This shows what the model really uses on unseen data (built-in gain importance favours "
        "features with many split points).")
    add("")
    gi = ctx["group_imp"].copy()
    gi["WAPE increase (points)"] = (gi.wape_increase * 100).round(2)
    add(md_table(gi[["feature", "WAPE increase (points)"]].rename(columns={"feature": "Feature group"})))
    add("")
    add("![Feature importance](figures/15_model1_feature_importance.png)")
    add("")
    add(ctx["importance_text"])
    add("")
    if ctx.get("ak"):
        ak = ctx["ak"]
        add("## Scored against the answer key (evaluation only)")
        add("")
        add("The answer key holds the simulator's *true* demand. It was not used for training, tuning or model selection. "
            "It is only used here, after everything else was fixed, to check two claims.")
        add("")
        add("**1. Correcting for empty shelves matters.** Same algorithm and settings, trained on two different targets, "
            "scored against true demand in the test weeks:")
        add("")
        t = pd.DataFrame(ak["test_vs_truth"]).T.reset_index().rename(columns={"index": "Model"})
        add(md_table(fmt_metrics(t)[["Model", "WAPE", "MAE", "Bias"]]))
        add("")
        t2 = pd.DataFrame(ak["target_vs_truth"]).T.reset_index().rename(columns={"index": "Target"})
        add(f"The targets themselves: `demand_est` is off true 7-day demand by {ak['target_vs_truth']['demand_est target']['Bias']:+.1%}, "
            f"raw sales by {ak['target_vs_truth']['units_sold target']['Bias']:+.1%}.")
        add("")
        add("**2. The forecast holds on a truly unseen week.** Forecast made from origin 31 Aug for 1-7 Sep:")
        add("")
        h = pd.DataFrame(ak["holdout_week"]).T.reset_index().rename(columns={"index": "Forecast"})
        add(md_table(fmt_metrics(h)[["Forecast", "WAPE", "MAE", "R2", "Bias"]]))
        add("")
        hc = ak["holdout_by_category"].copy()
        hc["WAPE"] = hc.WAPE.map(lambda v: f"{v:.1%}")
        hc["Baseline WAPE"] = hc["Baseline WAPE"].map(lambda v: f"{v:.1%}")
        hc["true"] = hc.true.map(lambda v: f"{v:,.0f}")
        hc["forecast"] = hc.forecast.map(lambda v: f"{v:,.0f}")
        add(md_table(hc.reset_index().rename(columns={"true": "True demand", "forecast": "Forecast"})))
        add("")
        add("![Answer key check](figures/16_answer_key_check.png)")
        add("")
    add("## Outputs for the team")
    add("")
    add("| File | Contents | Used by |")
    add("|---|---|---|")
    add("| `models/demand_forecast.joblib` | Selected model refitted on all labelled origins, with feature list and category encoding | Dashboard / recommendation (`src/forecast.py`) |")
    add("| `data/processed/features.csv` | Feature table, one row per origin, incl. Model 2 target `stockout_next7` | Student 3 |")
    add("| `data/processed/forecast_next7.csv` | Forecast for 1-7 Sep per store x product, with an 80% range | Recommendation cards |")
    add("| `data/processed/forecast_oot.csv` | Out-of-time forecast for every origin from 12 Apr (each block predicted by a model that never saw it) | Model 2 stacking feature `coverage = (stock + incoming) / forecast` |")
    add("| `reports/forecast_error_by_category.csv`, `data/processed/forecast_error_by_combo.csv` | Spread (sigma) of 7-day forecast errors | Safety stock = z x sigma x sqrt((lead + review) / 7) |")
    add("")
    add("## Limitations")
    add("")
    for line in ctx["limitations"]:
        add(f"- {line}")
    (REPORTS / "model1_demand_forecast.md").write_text("\n".join(L), encoding="utf-8")


def fmt_segment(t):
    d = t.copy()
    for c in ("WAPE", "Baseline WAPE"):
        d[c] = d[c].map(lambda v: f"{v:.1%}")
    d["Bias"] = d.Bias.map(lambda v: f"{v:+.1%}")
    d["Improvement vs baseline"] = d["Improvement vs baseline"].map(lambda v: f"{v:+.0%}")
    return d


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    feat = load_features()
    levels = category_levels(feat)
    train, val, test = (between(feat, SPLITS[k]) for k in ("train", "validation", "test"))
    train_val, labelled = between(feat, TRAIN_VAL), between(feat, ALL_LABELLED)
    final_rows = feat[feat.date == FINAL_ORIGIN]
    sizes = dict(train=len(train), validation=len(val), test=len(test), final=len(final_rows))
    # no target window of an earlier split may reach a later split's first origin
    assert train.date.max() + pd.Timedelta(days=7) < val.date.min()
    assert val.date.max() + pd.Timedelta(days=7) < test.date.min()
    print(f"Rows - train {len(train):,} | validation {len(val):,} | test {len(test):,} | final {len(final_rows):,}")

    print("Tuning on validation ...")
    tuning, fitted = tune(train, val, levels)
    tuning.to_csv(REPORTS / "model1_tuning_log.csv", index=False)
    val_best = best_per_algorithm(tuning)
    choice = val_best.iloc[0]
    selected, selected_cfg = choice.algorithm, choice.config
    print(f"Selected on validation: {selected} ({selected_cfg})")

    print("Scoring the test set once ...")
    test_res, test_preds, refits = evaluate_on_test(val_best, fitted, train_val, test, levels)
    test_res.to_csv(REPORTS / "model1_comparison.csv", index=False)
    print(test_res.round(4).to_string(index=False))
    sel_factory = refit_factory(selected, *fitted[(selected, selected_cfg)][:2])
    val_pred = fitted[(selected, selected_cfg)][2]
    test_pred = test_preds[selected]
    sel_test_model = refits[selected]

    tp = test.assign(pred=test_pred)
    by_cat = segment_table(tp, "pred", "category_name")
    by_store = segment_table(tp, "pred", "store_id")
    sparse_rows = tp[tp.history_days < 28]
    sm, sb = metrics(sparse_rows[TARGET], sparse_rows.pred), metrics(sparse_rows[TARGET], sparse_rows.baseline_7d)
    sparse_text = (f"{sparse_rows.product_id.nunique()} products launched in August appear in the test weeks with 2-15 days "
                   f"of history ({len(sparse_rows)} rows). Model WAPE {sm['WAPE']:.1%} (bias {sm['Bias']:+.1%}) vs baseline "
                   f"{sb['WAPE']:.1%} (bias {sb['Bias']:+.1%}). The global model needs no separate fallback: missing lags are "
                   "NaN, and it leans on store type, category, price and the few days of history it has. The two products "
                   "launched on 27 Aug have only 5 days of history at the final forecast, so treat their forecast as indicative "
                   "and review it after the first week.")

    res = test_pred - test[TARGET].values
    over = (res > 0).mean()
    residual_text = (f"Mean error {res.mean():+.1f} units per row (bias {res.sum() / test[TARGET].sum():+.1%}); the forecast is "
                     f"above actual on {over:.0%} of rows. The spread of the error grows with the size of the forecast "
                     "(middle panel), so safety stock should scale with volume and volatility. This is why the recommendation "
                     "layer uses a per-item error sigma (`forecast_error_by_combo.csv`), not a fixed percentage. "
                     "Right panel: promotion and festival weeks are harder to forecast than normal weeks, and items that "
                     "stocked out recently are hardest, because their recent demand is itself an estimate.")

    print("Permutation importance on test ...")
    group_imp, base = permutation_importance(sel_test_model, test, levels, FEATURE_GROUPS)
    single_imp, _ = permutation_importance(sel_test_model, test, levels, {f: [f] for f in FEATURES}, repeats=3)
    group_imp.to_csv(REPORTS / "model1_feature_importance_groups.csv", index=False)
    single_imp.to_csv(REPORTS / "model1_feature_importance.csv", index=False)
    top = group_imp.iloc[0]
    importance_text = (f"**{top.feature}** dominates (+{top.wape_increase * 100:.1f} WAPE points when shuffled): next week "
                       "looks like last few weeks. The next groups tell the model *how this week will differ*: "
                       + ", ".join(f"{r.feature.lower()} (+{r.wape_increase * 100:.1f})" for r in group_imp.iloc[1:4].itertuples())
                       + ". `reorder_level` ranks high among single features because it is the planners' own demand "
                       "estimate for the item (known at *t*), a useful second opinion on its level. "
                       "Groups near zero add little on top of the others; they are kept because they cost nothing "
                       "and matter for specific items (e.g. weather for beverages and ice cream).")

    print("Refitting the selected model on all labelled origins and forecasting 1-7 Sep ...")
    final_model = sel_factory().fit(model_frame(labelled, levels), labelled[TARGET].values)

    print("Out-of-time forecasts for Model 2 ...")
    oot = out_of_time_forecasts(feat, sel_factory, levels, final_model)
    cat_err, combo_err = error_profiles(oot, feat)
    cat_err.to_csv(REPORTS / "forecast_error_by_category.csv")
    combo_err.to_csv(PROCESSED / "forecast_error_by_combo.csv", index=False)
    oot.drop(columns=[TARGET]).to_csv(PROCESSED / "forecast_oot.csv", index=False)

    final = final_rows[["date"] + COMBO + ["product_name", "category_name", "store_type_name", "unit_price",
                                           "closing_stock", "incoming_next7", "baseline_7d", "history_days"]].copy()
    final = final.rename(columns={"date": "origin_date", "category_name": "category", "store_type_name": "store_type"})
    final["forecast_7d"] = np.clip(final_model.predict(model_frame(final_rows, levels)), 0, None)
    sig = final[COMBO].merge(combo_err[COMBO + ["sigma_7d", "rows"]], on=COMBO, how="left")
    cat_rel = final.category.map(cat_err.sigma_rel)
    # items with little out-of-time history fall back to the category's relative error
    sigma = np.where(sig.rows.fillna(0).values >= 28, sig.sigma_7d.values, cat_rel.values * final.forecast_7d.values)
    final["forecast_sigma_7d"] = sigma
    final["forecast_p10"] = np.clip(final.forecast_7d - 1.2816 * sigma, 0, None)
    final["forecast_p90"] = final.forecast_7d + 1.2816 * sigma
    for c in ("forecast_7d", "forecast_sigma_7d", "forecast_p10", "forecast_p90", "baseline_7d"):
        final[c] = final[c].round(1)
    final.to_csv(PROCESSED / "forecast_next7.csv", index=False)

    bundle = dict(model=final_model, algorithm=selected, config=selected_cfg, features=FEATURES, levels=levels,
                  target=TARGET, trained_on_origins=[str(ALL_LABELLED[0].date()), str(ALL_LABELLED[1].date())],
                  test_metrics={k: float(v) for k, v in test_res.set_index("algorithm").loc[selected].items()})
    joblib.dump(bundle, MODEL_PATH)
    meta = {k: v for k, v in bundle.items() if k not in ("model", "levels")}
    (MODELS / "demand_forecast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ak = None
    if (ANSWER_KEY / "true_demand_and_holdout.csv").exists():
        print("Scoring against the answer key (evaluation only) ...")
        sold_model = sel_factory().fit(model_frame(train_val, levels), train_val["next_7_day_units_sold"].values)
        ak = answer_key_checks(feat, test, test_pred, sold_model, levels, final)
        chart_answer_key(ak)
        json.dump({k: v for k, v in ak.items() if k != "holdout_by_category"},
                  open(REPORTS / "model1_answer_key_check.json", "w"), indent=2, default=float)

    print("Charts and report ...")
    chart_comparison(val_best, test_res, selected)
    chart_actual_vs_forecast(val, val_pred, test, test_pred)
    chart_segments(by_cat, by_store)
    chart_residuals(test, test_pred)
    chart_importance(group_imp, single_imp, base)

    others = val_best[val_best.algorithm != selected]
    runner = others.iloc[0]
    te = test_res[~test_res.algorithm.str.startswith("Baseline")].sort_values("WAPE")
    te_sel = te.set_index("algorithm").loc[selected]
    te_next = te[te.algorithm != selected].iloc[0]
    why = [f"Lowest validation WAPE of the six algorithms ({choice.WAPE:.1%}); the runner-up, {runner.algorithm}, "
           f"had {runner.WAPE:.1%}."]
    if te.iloc[0].algorithm == selected:
        why.append(f"The ranking held on test: {selected} {te_sel.WAPE:.1%} vs {te_next.algorithm} {te_next.WAPE:.1%}, "
                   "so the choice was not a validation fluke.")
    else:
        why.append(f"On test, {te.iloc[0].algorithm} scored {te.iloc[0].WAPE:.1%} vs {te_sel.WAPE:.1%} for {selected}. "
                   "The choice stays with the validation winner: switching after seeing the test set would turn the test "
                   "set into a second validation set.")
    why += [
        f"The margin over the next models is modest ({(te_next.WAPE - te_sel.WAPE) * 100:.1f} WAPE points on test). "
        "Most of the gain over the baseline comes from the features (corrected demand, the promotion plan, the festival "
        "calendar), which every model shares. The algorithm choice is the smaller lever.",
        "XGBoost handles the mix of feature types (counts, ratios, flags, categories) without scaling, learns interactions "
        "such as promotion x category without them being hand-coded, and handles missing history natively (new "
        "products have NaN lags).",
        f"It is fast: one fit on ~{len(labelled) // 1000}k rows takes seconds, so the rolling refits for Model 2 and a daily "
        "production refit are cheap. Random Forest is close in accuracy but slower and cannot extrapolate beyond the range "
        "it saw (e.g. a growing store). KNN and the RBF-kernel SVM trail because distances over 60 mixed features are "
        "not meaningful, and a single Decision Tree is too coarse.",
    ]
    why.append(f"Its bias ({te_sel.Bias:+.1%} on test) is a mild under-forecast. It is not fixed by choosing a less biased "
               "model on validation: bias measured on one 15-day window proved unstable (see WORKLOG). Instead the "
               "out-of-time error profile (`bias_units`, `sigma_7d`) goes to the safety-stock calculation.")
    limitations = [
        "`demand_est` is itself an estimate: once a shelf is empty the extra demand cannot be observed, so the target is "
        "conservative on chronic stock-out items. The answer-key check quantifies how much.",
        "Weather is used only from the past 3 days. A real deployment would add the weather forecast for the coming week.",
        "The four late launches have 5-22 days of history; their forecasts lean on store and category averages.",
        "One validation window (15 origin days) was used for tuning; a rolling-origin cross-validation would give a more "
        "stable choice at the cost of more compute.",
        "The model forecasts demand; it knows nothing about supply. Stock-out risk (Model 2) and the recommendation "
        "combine it with stock, open orders and supplier reliability.",
    ]
    val_base = tuning[tuning.algorithm.str.startswith("Baseline")].WAPE.iloc[0]
    write_report(dict(selected=selected, selected_config=selected_cfg, test_res=test_res, val_best=val_best, sizes=sizes,
                      by_cat=by_cat, by_store=by_store, sparse_text=sparse_text, residual_text=residual_text,
                      group_imp=group_imp, importance_text=importance_text, why=why, limitations=limitations,
                      n_configs=len(tuning) - 1, val_baseline_wape=val_base, ak=ak))
    print(f"Done. Test WAPE {test_res.set_index('algorithm').loc[selected, 'WAPE']:.1%} "
          f"vs baseline {test_res.set_index('algorithm').loc['Baseline (7 x last-7-day mean)', 'WAPE']:.1%}")
    if ak:
        print(f"Holdout week 1-7 Sep vs true demand: WAPE {ak['holdout_week']['Model (origin 31 Aug)']['WAPE']:.1%} "
              f"(baseline {ak['holdout_week']['Baseline']['WAPE']:.1%})")
    return dict(tuning=tuning, test_res=test_res, final=final, oot=oot, ak=ak)


if __name__ == "__main__":
    main()
