"""
StockSense dashboard/data_access.py
Cached data loaders for the Streamlit dashboard.
All paths are relative to the repo root (resolved from this file's location).
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"
FIGURES   = REPORTS / "figures"
MODELS    = ROOT / "models"

# Make src/ importable
sys.path.insert(0, str(ROOT / "src"))


@st.cache_data
def load_master() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_forecast_oot() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "forecast_oot.csv", parse_dates=["date"])


@st.cache_data
def load_forecast_next7() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "forecast_next7.csv")
    df["date"] = pd.to_datetime(df["origin_date"])
    return df


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    p = PROCESSED / "recommendations_1_7_sep.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_model1_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "model1_comparison.csv")


@st.cache_data
def load_model2_comparison() -> pd.DataFrame:
    p = REPORTS / "model2_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    p = REPORTS / "model2_feature_importance_groups.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_error_by_category() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "forecast_error_by_category.csv")


@st.cache_data
def load_kpi_summary() -> pd.DataFrame:
    p = REPORTS / "kpi_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_resource
def load_stockout_model():
    from stockout import load_model
    return load_model()


@st.cache_resource
def load_demand_model():
    from forecast import load_model
    return load_model()


def figure_path(num: int) -> Path | None:
    """Return the path to figure {num}_*.png if it exists."""
    matches = sorted(FIGURES.glob(f"{num:02d}_*.png"))
    return matches[0] if matches else None
