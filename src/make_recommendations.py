"""
StockSense - Round 3 Phase 3 (Student 3: Decision Intelligence)
src/make_recommendations.py - Build the final recommendation table for Sep 1-7

Takes the final forecast (origin 2026-08-31 from forecast_next7.csv),
joins with features.csv rows for date==2026-08-31, builds Model 2 stacked
features, predicts calibrated stock-out probabilities, applies the business
action layer, and writes data/processed/recommendations_1_7_sep.csv (387 rows).

Run from repo root:
    python src/make_recommendations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockout import load_model, predict_proba, risk_level as so_risk_level
from recommend import build_recommendation, get_sigma_7d
from explain import explain_row, top_reasons

PROCESSED = ROOT / "data" / "processed"
ORIGIN_DATE = pd.Timestamp("2026-08-31")


def build_model2_frame(feat: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Join features (origin=Aug 31) with the Sep 1-7 forecast and add stacked features.
    forecast_next7.csv uses 'origin_date' column (not 'date').
    """
    # Rename forecast date column to match features join key
    fcast = forecast.copy()
    fcast["date"] = pd.to_datetime(fcast["origin_date"])

    # Take only Aug 31 origin rows from features
    feat31 = feat[feat.date == ORIGIN_DATE].copy()
    print(f"  Feature rows for origin {ORIGIN_DATE.date()}: {len(feat31)}")

    # Merge on (date, store_id, product_id)
    df = feat31.merge(
        fcast[["date", "store_id", "product_id",
               "forecast_7d", "forecast_sigma_7d", "forecast_p10", "forecast_p90",
               "closing_stock", "incoming_next7", "unit_price"]],
        on=["date", "store_id", "product_id"], how="inner",
        suffixes=("_feat", "_fcast"),
    )
    print(f"  Merged rows: {len(df)}")

    # Prefer the forecast file's stock/incoming/price (more authoritative for Sep)
    for col in ["closing_stock", "incoming_next7", "unit_price"]:
        fcast_col = col + "_fcast"
        feat_col  = col + "_feat"
        if fcast_col in df.columns:
            df[col] = df[fcast_col].fillna(df[feat_col] if feat_col in df.columns else np.nan)

    # Build stacked features (identical formulas to train_stockout.py)
    df["coverage"]        = (df.closing_stock + df.incoming_next7) / df.forecast_7d.clip(lower=1)
    df["forecast_vs_rm7"] = df.forecast_7d / (df.rolling_mean_7 * 7).clip(lower=1)
    df["lead_cover_gap"]  = (
        df.closing_stock + df.incoming_next7
        - df.forecast_7d * (df.effective_lead_time / 7)
    )

    return df


def main():
    print("=" * 65)
    print("make_recommendations.py - Sep 1-7 action table")
    print("=" * 65)

    print("\nLoading model ...")
    bundle = load_model()

    print("Loading data ...")
    feat     = pd.read_csv(PROCESSED / "features.csv",     parse_dates=["date"])
    forecast = pd.read_csv(PROCESSED / "forecast_next7.csv")

    print("\nBuilding model-2 frame for origin Aug 31 ...")
    df = build_model2_frame(feat, forecast)

    print("\nPredicting stock-out probabilities ...")
    probs = predict_proba(bundle, df)
    df["stockout_probability"] = probs
    df["risk_level"] = so_risk_level(probs, bundle)

    # Pre-compute feature medians (numeric only) for the what-if flip explainer
    feat_medians = {
        f: df[f].median()
        for f in bundle["features"]
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    }

    print("Computing top reasons and building recommendation table ...")
    rows = []
    for _, row in df.iterrows():
        reasons = top_reasons(row, bundle, n=4, feat_medians=feat_medians)
        rec = build_recommendation(row, float(row["stockout_probability"]),
                                   top_reasons_str=reasons)
        # Add store_type for filtering
        rec["store_type"] = str(row.get("store_type_name", row.get("store_type", "")))
        rows.append(rec)

    rec_df = pd.DataFrame(rows)

    # Sort by expected_loss_inr descending (D3.6)
    rec_df = rec_df.sort_values("expected_loss_inr", ascending=False).reset_index(drop=True)

    out_path = PROCESSED / "recommendations_1_7_sep.csv"
    rec_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.relative_to(ROOT)}")
    print(f"  Rows: {len(rec_df)}")
    print(f"  HIGH risk items: {(rec_df.risk_level == 'HIGH').sum()}")
    print(f"  MEDIUM risk items: {(rec_df.risk_level == 'MEDIUM').sum()}")
    print(f"  LOW risk items: {(rec_df.risk_level == 'LOW').sum()}")
    print(f"  Items with reorder>0: {(rec_df.recommended_order > 0).sum()}")
    total_loss = rec_df.expected_loss_inr.sum()
    print(f"  Total expected loss at risk: Rs {total_loss:,.0f}")

    print("\n--- TOP 10 ACTIONS BY EXPECTED LOSS ---")
    top10 = rec_df.head(10)[[
        "store", "product", "category", "risk_level",
        "current_stock", "forecast_7d", "stockout_probability",
        "recommended_order", "expected_loss_inr", "manager_action"
    ]]
    print(top10.to_string(index=False))

    print("\n" + "=" * 65)
    print("PHASE 3 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
