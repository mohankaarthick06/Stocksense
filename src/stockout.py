"""
StockSense - Round 3 Phase 2 (Student 3: Decision Intelligence)
src/stockout.py - Load/predict helpers for the saved stock-out risk model.

Mirrors src/forecast.py style so the dashboard and make_recommendations.py
can import both consistently:

    from stockout import load_model, predict_proba
    bundle = load_model()
    df["stockout_prob"] = predict_proba(bundle, df)

Run: python src/stockout.py  (self-test)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
MODELS     = ROOT / "models"
MODEL_PATH = MODELS / "stockout_model.joblib"


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the stockout model bundle saved by src/train_stockout.py."""
    return joblib.load(path)


def _encode(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Encode categoricals with the saved level mapping and return float matrix."""
    feat_cols   = bundle["features"]
    categorical = bundle["categorical"]
    levels      = bundle["levels"]

    X = df[feat_cols].copy()
    for c in categorical:
        codes = pd.Categorical(X[c], categories=levels[c]).codes.astype(float)
        X[c] = np.where(codes < 0, np.nan, codes)
    return X.astype(float)


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Calibrated stock-out probability for every row in df.

    df must contain all columns in bundle['features'] (including the three
    stacked features: coverage, forecast_vs_rm7, lead_cover_gap).
    """
    X = _encode(df, bundle)
    return bundle["model"].predict_proba(X)[:, 1]


def risk_level(prob: float | np.ndarray, bundle: dict) -> str | np.ndarray:
    """Map calibrated probability to HIGH / MEDIUM / LOW."""
    thr_high   = bundle.get("thr_high",   0.70)
    thr_medium = bundle.get("thr_medium", 0.40)

    if np.ndim(prob) == 0:
        if prob >= thr_high:
            return "HIGH"
        elif prob >= thr_medium:
            return "MEDIUM"
        return "LOW"

    out = np.full(len(prob), "LOW", dtype=object)
    out[prob >= thr_medium] = "MEDIUM"
    out[prob >= thr_high]   = "HIGH"
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading stockout model ...")
    bundle = load_model()
    print(f"  Algorithm  : {bundle['cal_method']} calibrated model")
    print(f"  Features   : {len(bundle['features'])} features")
    print(f"  Thresholds : HIGH>={bundle['thr_high']}  MEDIUM>={bundle['thr_medium']}")
    meta_path = MODELS / "stockout_model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        tm = meta.get("test_metrics", {})
        print(f"  Test PR-AUC: {tm.get('pr_auc', 'n/a'):.4f}")
        print(f"  Test Brier : {tm.get('brier', 'n/a'):.4f}")
    print("src/stockout.py: OK")
