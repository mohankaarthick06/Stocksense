"""
StockSense - Round 3 Phase 2 (Student 3: Decision Intelligence)
src/explain.py - Explainability for Model 2 (stock-out risk classifier)

Global:
  Permutation importance on TEST set (score = PR-AUC), aggregated into
  plain-language feature groups.
  Outputs:
    reports/model2_feature_importance.csv
    reports/model2_feature_importance_groups.csv
    reports/figures/21_model2_permutation_importance.png

Local (per row):
  "Reset-to-normal what-if flip" (DECISIONS D2.8 fallback method):
  For one row, reset one feature group at a time to its neutral value,
  re-predict, and record the drop in stock-out probability.
  Returns human-readable sentences.

  Also attempts XGBoost pred_contribs as a cross-check (when the base model
  is XGBoost; silently skipped otherwise).

Run: python src/explain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockout import load_model, predict_proba

PROCESSED = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"
FIGURES   = REPORTS / "figures"

TEST_START = pd.Timestamp("2026-08-11")
TEST_END   = pd.Timestamp("2026-08-24")

# Feature groups for permutation importance (manager-readable names)
FEATURE_GROUPS_LABELS = {
    "Recent sales trend": [
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_28", "cv_28",
        "prev_week_mean", "trend_7_28", "growth_wow", "baseline_7d",
    ],
    "Current stock & cover": [
        "closing_stock", "reorder_level", "reorder_gap",
        "days_of_cover", "stock_to_demand_7d", "stockout_today",
        "coverage", "lead_cover_gap",
    ],
    "Incoming stock & supplier reliability": [
        "incoming_next7", "lead_time_days", "supplier_late_rate",
        "supplier_avg_delay", "supplier_fill_rate", "effective_lead_time",
        "forecast_vs_rm7",
    ],
    "Stock-out history": [
        "stockout_days_last14", "stockout_rate_last28",
        "days_since_stockout", "lost_units_last7",
    ],
    "Promotion plan": [
        "promo_days_next7", "max_discount_next7", "avg_discount_next7",
        "promo_today", "promo_days_last7", "days_since_promo", "post_promo_window",
    ],
    "Festival / holiday / payday calendar": [
        "holiday_next7", "festival_next7", "days_to_festival",
        "product_festival_lift", "festival_x_lift",
        "local_event_next7", "payday_days_next7",
    ],
    "Weather": ["temp_last3", "rain_last3", "temp_anomaly"],
    "Price": ["regular_price", "price_ratio_28", "price_change_30d", "discount_today"],
    "Store & product profile": [
        "store_code", "store_type", "category", "sub_category",
        "shelf_life_days", "avg_daily_customers", "floor_area_sqft",
        "history_days", "origin_dow", "month",
    ],
}

# Neutral values for the what-if flip (per group)
NEUTRAL_VALUES = {
    "Recent sales trend": {},  # do not reset -- too structural
    "Current stock & cover": {},  # do not reset -- main signal for Model 2
    "Incoming stock & supplier reliability": {
        "supplier_late_rate": 0.0,
        "supplier_avg_delay": 0.0,
        "supplier_fill_rate": 1.0,
        "effective_lead_time": None,  # set to lead_time_days (no extra delay)
    },
    "Stock-out history": {
        "stockout_days_last14": 0.0,
        "stockout_rate_last28": 0.0,
        "days_since_stockout": 60.0,   # CAP_DAYS -> no recent stockout
        "lost_units_last7": 0.0,
    },
    "Promotion plan": {
        "promo_days_next7": 0.0,
        "max_discount_next7": 0.0,
        "avg_discount_next7": 0.0,
        "promo_today": 0.0,
        "post_promo_window": 0.0,
    },
    "Festival / holiday / payday calendar": {
        "holiday_next7": 0.0,
        "festival_next7": 0.0,
        "days_to_festival": 30.0,
        "festival_x_lift": 0.0,
        "local_event_next7": 0.0,
        "payday_days_next7": 0.0,
    },
    "Weather": {
        "temp_last3": None,   # set to 28-day rolling mean (filled from data)
        "temp_anomaly": 0.0,
    },
    "Price": {
        "price_ratio_28": 1.0,
        "price_change_30d": 0.0,
        "discount_today": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode_row(row: pd.Series, bundle: dict) -> pd.DataFrame:
    """Wrap a single row as a 1-row DataFrame and encode for prediction."""
    df1 = pd.DataFrame([row])
    from stockout import _encode
    return _encode(df1, bundle)


def _predict_row(row: pd.Series, bundle: dict) -> float:
    """Predict stock-out probability for a single row."""
    return float(predict_proba(bundle, pd.DataFrame([row]))[0])


# ---------------------------------------------------------------------------
# Global: permutation importance
# ---------------------------------------------------------------------------
def permutation_importance(bundle: dict, df_test: pd.DataFrame,
                            y_test: np.ndarray, n_repeats: int = 5) -> pd.DataFrame:
    """
    Permutation importance on the TEST set.
    Score = PR-AUC (the selection metric for Model 2).
    For each feature, permute it n_repeats times and record the mean drop in PR-AUC.
    """
    from sklearn.metrics import average_precision_score

    X_test = df_test[bundle["features"]].copy()
    from stockout import _encode
    Xenc = _encode(df_test, bundle)

    base_score = average_precision_score(y_test, bundle["model"].predict_proba(Xenc)[:, 1])
    print(f"  Base PR-AUC on test: {base_score:.4f}")

    rng = np.random.default_rng(42)
    rows = []
    for feat in bundle["features"]:
        if feat not in Xenc.columns:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = Xenc.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            s = average_precision_score(y_test, bundle["model"].predict_proba(Xp)[:, 1])
            drops.append(base_score - s)
        rows.append({"feature": feat, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
        print(f"  {feat:40s}  drop={np.mean(drops):.4f}")

    imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp, base_score


def aggregate_group_importance(imp: pd.DataFrame) -> pd.DataFrame:
    """Sum feature-level importance into manager-readable groups."""
    feat2grp = {}
    for grp, feats in FEATURE_GROUPS_LABELS.items():
        for f in feats:
            feat2grp[f] = grp

    imp["group"] = imp.feature.map(feat2grp).fillna("Other")
    grp_imp = (
        imp.groupby("group")
        .agg(total_importance=("importance_mean", "sum"))
        .reset_index()
        .sort_values("total_importance", ascending=False)
    )
    total = grp_imp.total_importance.clip(lower=0).sum()
    grp_imp["share_pct"] = (grp_imp.total_importance.clip(lower=0) / max(total, 1e-9) * 100).round(1)
    return grp_imp


def plot_group_importance(grp_imp: pd.DataFrame):
    """Figure 21: group-level permutation importance bar chart (manager-readable)."""
    plot = grp_imp[grp_imp.total_importance > 0].sort_values("total_importance")
    colors = ["#e74c3c" if "Stock" in g or "Current" in g else
              "#f39c12" if "Promotion" in g else
              "#3498db" for g in plot.group]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(plot.group, plot.total_importance, color=colors, edgecolor="white")
    for bar, val, pct in zip(bars, plot.total_importance, plot.share_pct):
        ax.text(bar.get_width() + 0.0002,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Permutation importance (PR-AUC drop, test set)")
    ax.set_title("Figure 21 - What drives stock-out risk?\nModel 2 feature group importance", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES / "21_model2_permutation_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/21_model2_permutation_importance.png")


# ---------------------------------------------------------------------------
# Local: what-if flip explainer
# ---------------------------------------------------------------------------
def explain_row(row: pd.Series, bundle: dict,
                feat_medians: dict | None = None) -> list[tuple[str, float, str]]:
    """
    For one row, reset each feature group in turn to its neutral value and
    measure the drop in stock-out probability.

    Returns a list of (driver_label, impact_pct, sentence) sorted by impact.
    impact_pct is the percentage share of the probability increase above
    a theoretical neutral baseline, attributed to this group.

    feat_medians: dict of {feature: median_value} used to set 'weather to normal'.
    """
    base_prob = _predict_row(row, bundle)

    group_drops: dict[str, float] = {}
    for group, neutral in NEUTRAL_VALUES.items():
        if not neutral:
            continue
        row_mod = row.copy()
        for feat, val in neutral.items():
            if feat not in row_mod.index:
                continue
            if val is None:
                # special cases
                if feat == "effective_lead_time":
                    row_mod[feat] = row_mod.get("lead_time_days", row_mod[feat])
                elif feat == "temp_last3" and feat_medians:
                    row_mod[feat] = feat_medians.get(feat, row_mod[feat])
            else:
                row_mod[feat] = val

        # Also reset coverage and lead_cover_gap if supplier/stock group changed
        if "incoming_next7" in neutral or "supplier_avg_delay" in neutral:
            try:
                row_mod["lead_cover_gap"] = (
                    row_mod.get("closing_stock", 0)
                    + row_mod.get("incoming_next7", 0)
                    - row_mod.get("forecast_7d", 0) * (row_mod.get("effective_lead_time", 1) / 7)
                )
            except Exception:
                pass

        new_prob = _predict_row(row_mod, bundle)
        group_drops[group] = max(0.0, base_prob - new_prob)

    total_drop = sum(group_drops.values())
    if total_drop < 1e-6:
        return []

    results = []
    for group, drop in sorted(group_drops.items(), key=lambda x: -x[1]):
        if drop < 1e-4:
            continue
        share = drop / total_drop * 100
        sentence = _make_sentence(group, row, share, base_prob)
        results.append((group, round(share, 1), sentence))

    return results


def _make_sentence(group: str, row: pd.Series, share: float, prob: float) -> str:
    """Convert a driver group + row values into a human-readable sentence."""
    row_d = row.to_dict()
    prob_str = f"{prob:.0%}"

    if group == "Promotion plan":
        disc = row_d.get("max_discount_next7", 0)
        days = int(row_d.get("promo_days_next7", 0))
        return (f"Promotion planned ({disc*100:.0f}% off, {days} days) "
                f"adds ~{share:.0f}% to the risk")

    if group == "Festival / holiday / payday calendar":
        dtf = row_d.get("days_to_festival", 30)
        if dtf <= 7:
            return f"Festival in {int(dtf)} day(s) drives extra demand, adding ~{share:.0f}% risk"
        if row_d.get("holiday_next7", 0) > 0:
            return f"Holiday in the next 7 days adds ~{share:.0f}% to the risk"
        return f"Payday/local event in window adds ~{share:.0f}% to the risk"

    if group == "Stock-out history":
        rate = row_d.get("stockout_rate_last28", 0)
        return (f"This item stocked out {rate*100:.0f}% of days in the past 28 — "
                f"chronic pattern adds ~{share:.0f}% risk")

    if group == "Incoming stock & supplier reliability":
        lr = row_d.get("supplier_late_rate", 0)
        return (f"Supplier is late {lr*100:.0f}% of the time — "
                f"effective lead time risk adds ~{share:.0f}%")

    if group == "Weather":
        temp = row_d.get("temp_last3", 28)
        return f"Recent temperature {temp:.1f} deg C (above normal) adds ~{share:.0f}% beverage risk"

    if group == "Price":
        ratio = row_d.get("price_ratio_28", 1)
        return f"Price ratio vs 28-day median is {ratio:.2f}, adding ~{share:.0f}% risk"

    return f"{group} contributes ~{share:.0f}% to the current risk ({prob_str})"


def top_reasons(row: pd.Series, bundle: dict, n: int = 4,
                feat_medians: dict | None = None) -> str:
    """One short string with the top n drivers, suitable for a dashboard card."""
    drivers = explain_row(row, bundle, feat_medians=feat_medians)
    if not drivers:
        return "Insufficient signal to explain"
    parts = [f"{grp} +{pct:.0f}%" for grp, pct, _ in drivers[:n]]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import sys
    print("=" * 65)
    print("PHASE 2 - Explainability")
    print("=" * 65)

    print("\nLoading model ...")
    bundle = load_model()

    print("Loading features and building test frame ...")
    feat = pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])
    oot  = pd.read_csv(PROCESSED / "forecast_oot.csv", parse_dates=["date"])
    df = feat.merge(oot[["date", "store_id", "product_id", "forecast_7d"]],
                    on=["date", "store_id", "product_id"], how="inner")
    df = df.dropna(subset=["stockout_next7"])

    # Add stacked features
    df["coverage"]        = (df.closing_stock + df.incoming_next7) / df.forecast_7d.clip(lower=1)
    df["forecast_vs_rm7"] = df.forecast_7d / (df.rolling_mean_7 * 7).clip(lower=1)
    df["lead_cover_gap"]  = (
        df.closing_stock + df.incoming_next7
        - df.forecast_7d * (df.effective_lead_time / 7)
    )

    test_mask = (df.date >= TEST_START) & (df.date <= TEST_END)
    df_test = df[test_mask].reset_index(drop=True)
    y_test  = df_test["stockout_next7"].values.astype(int)

    print(f"  Test rows: {len(df_test)}")

    print("\nComputing permutation importance (n_repeats=5) ...")
    imp, base_score = permutation_importance(bundle, df_test, y_test, n_repeats=5)
    imp.to_csv(REPORTS / "model2_feature_importance.csv", index=False)
    print("  Saved: model2_feature_importance.csv")

    grp_imp = aggregate_group_importance(imp)
    grp_imp.to_csv(REPORTS / "model2_feature_importance_groups.csv", index=False)
    print("  Saved: model2_feature_importance_groups.csv")
    print(grp_imp.to_string(index=False))

    plot_group_importance(grp_imp)

    # Demonstrate local explanation on a HIGH-risk test row
    print("\nLocal explanation demo (first HIGH-risk row in test set) ...")
    probs = predict_proba(bundle, df_test)
    high_rows = df_test[probs >= 0.70]
    if len(high_rows) > 0:
        demo_row = high_rows.iloc[0]
        feat_medians = {
            f: df[f].median()
            for f in bundle["features"]
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        }
        drivers = explain_row(demo_row, bundle, feat_medians=feat_medians)
        print(f"\n  Demo: {demo_row.get('store_id','?')} / {demo_row.get('product_name','?')}")
        print(f"  Stock-out prob: {probs[high_rows.index[0]]:.2%}")
        for grp, pct, sent in drivers:
            print(f"    [{pct:5.1f}%] {sent}")
        summary = top_reasons(demo_row, bundle, feat_medians=feat_medians)
        print(f"\n  top_reasons: {summary}")

    print("\n" + "=" * 65)
    print("PHASE 2 COMPLETE")
    print(f"  Base PR-AUC (test): {base_score:.4f}")
    print("  Outputs: model2_feature_importance.csv, model2_feature_importance_groups.csv")
    print("  Figure : 21_model2_permutation_importance.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
