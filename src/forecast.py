"""
StockSense - shared helpers to load the saved demand model and forecast with it.

Used by the training script, and by the dashboard / recommendation layer:

    from forecast import load_model, predict
    bundle = load_model()
    features = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
    today = features[features.date == features.date.max()]
    today["forecast_7d"] = predict(bundle, today)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import CATEGORICAL, FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "demand_forecast.joblib"


def category_levels(df):
    """Fixed category -> code mapping, saved with the model so new data is encoded the same way."""
    return {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL}


def model_frame(df, levels):
    """Feature matrix in model order; categorical features as integer codes (-1 -> NaN for unseen)."""
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        codes = pd.Categorical(X[c], categories=levels[c]).codes.astype(float)
        X[c] = np.where(codes < 0, np.nan, codes)
    return X.astype(float)


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(bundle, df):
    """7-day demand forecast (units) for every row of a features table."""
    return np.clip(bundle["model"].predict(model_frame(df, bundle["levels"])), 0, None)
