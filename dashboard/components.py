"""
StockSense dashboard/components.py
Reusable Plotly charts and styled cards for the dashboard.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RISK_COLORS = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
PALETTE     = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
               "#1abc9c", "#e67e22", "#34495e"]


def risk_badge(risk: str) -> str:
    """HTML badge for a risk level."""
    col = RISK_COLORS.get(risk, "#95a5a6")
    return (f'<span style="background:{col};color:white;padding:3px 9px;'
            f'border-radius:12px;font-weight:700;font-size:0.85em;">{risk}</span>')


def kpi_card(label: str, value: str, delta: str = "", color: str = "#3498db"):
    """Render a single KPI card using st.metric."""
    st.metric(label=label, value=value, delta=delta if delta else None)


def risk_donut(rec_df: pd.DataFrame) -> go.Figure:
    """Donut chart of HIGH / MEDIUM / LOW counts."""
    counts = rec_df.risk_level.value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker_colors=[RISK_COLORS[r] for r in counts.index],
        textinfo="label+value",
        hovertemplate="%{label}: %{value} items (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="Stock-out risk distribution (Sep 1-7)",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=50, b=30, l=0, r=0),
        height=320,
    )
    return fig


def revenue_trend(master: pd.DataFrame) -> go.Figure:
    """Weekly revenue trend line."""
    weekly = (master.assign(week=master.date.dt.to_period("W").dt.start_time)
              .groupby("week").agg(revenue=("revenue", "sum")).reset_index())
    fig = px.line(weekly, x="week", y="revenue", markers=True,
                  title="Weekly Revenue (Rs)",
                  labels={"week": "Week", "revenue": "Revenue (Rs)"},
                  color_discrete_sequence=["#3498db"])
    fig.update_layout(height=320, margin=dict(t=50, b=30))
    return fig


def stock_heatmap(rec_df: pd.DataFrame) -> go.Figure:
    """Store x category heatmap of avg stock-out probability."""
    pivot = (rec_df.groupby(["store", "category"])
             .stockout_probability.mean()
             .unstack(fill_value=0))
    fig = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        zmin=0, zmax=1,
        title="Average stock-out probability by store x category",
        labels={"color": "Prob"},
        text_auto=".2f",
    )
    fig.update_layout(height=400, margin=dict(t=60, b=30))
    return fig


def actual_vs_forecast_chart(oot: pd.DataFrame, feat: pd.DataFrame,
                              store_id: str, product_id: str) -> go.Figure:
    """Actual vs OOT forecast for a given store x product (test origins + Sep forecast)."""
    sub_oot  = oot[(oot.store_id == store_id) & (oot.product_id == product_id)].copy()
    sub_feat = feat[(feat.store_id == store_id) & (feat.product_id == product_id)].copy()

    merged = sub_oot.merge(
        sub_feat[["date", "next_7_day_demand"]],
        on="date", how="left"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged.date, y=merged.next_7_day_demand,
        name="Actual (7-day demand)", mode="lines+markers",
        line=dict(color="#2ecc71", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=merged.date, y=merged.forecast_7d,
        name="OOT Forecast", mode="lines",
        line=dict(color="#3498db", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"Actual vs Forecast — {store_id} / {product_id}",
        xaxis_title="Origin date", yaxis_title="7-day demand (units)",
        height=380, margin=dict(t=60, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def wape_by_category(err_df: pd.DataFrame) -> go.Figure:
    """Bar chart of forecast WAPE by category."""
    df = err_df.copy()
    df["wape_pct"] = df.get("sigma_rel", df.get("wape", 0)) * 100 if "sigma_rel" in df.columns else 0
    fig = px.bar(
        df.sort_values("wape_pct", ascending=True),
        x="wape_pct", y="category_name", orientation="h",
        title="Forecast error (sigma_rel) by category",
        labels={"wape_pct": "Sigma relative (%)", "category_name": "Category"},
        color="wape_pct",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    fig.update_layout(height=360, margin=dict(t=60, b=30), coloraxis_showscale=False)
    return fig


def feature_importance_bar(grp_imp: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of group-level permutation importance."""
    df = grp_imp[grp_imp.total_importance > 0].sort_values("total_importance")
    fig = px.bar(
        df, x="total_importance", y="group", orientation="h",
        text="share_pct",
        title="What drives stock-out risk? (permutation importance, test set)",
        labels={"total_importance": "PR-AUC drop", "group": "Feature group"},
        color="total_importance",
        color_continuous_scale=["#3498db", "#e74c3c"],
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(height=380, margin=dict(t=60, b=30), coloraxis_showscale=False)
    return fig


def recommendation_card(row: pd.Series):
    """Render an expandable recommendation card for one store x product."""
    risk  = str(row.get("risk_level", "LOW"))
    store = str(row.get("store", ""))
    prod  = str(row.get("product", ""))
    badge = risk_badge(risk)

    with st.expander(
        f"{'🔴' if risk=='HIGH' else '🟡' if risk=='MEDIUM' else '🟢'} "
        f"{store} — {prod} | {risk}", expanded=(risk == "HIGH")
    ):
        c1, c2, c3 = st.columns(3)
        c1.metric("Current stock",    str(int(row.get("current_stock", 0))),
                  delta=None)
        c1.metric("Incoming (7d)",    str(int(row.get("incoming_stock", 0))))
        c2.metric("Forecast 7d",
                  f"{row.get('forecast_7d', 0):.0f} units",
                  delta=f"P10={row.get('forecast_p10',0):.0f} / P90={row.get('forecast_p90',0):.0f}")
        c2.metric("Stock-out prob",   f"{row.get('stockout_probability', 0):.0%}")
        c3.metric("Recommended order", str(int(row.get("recommended_order", 0))))
        c3.metric("Expected loss",
                  f"Rs {row.get('expected_loss_inr', 0):,.0f}")

        st.markdown(f"**Risk level:** {badge}", unsafe_allow_html=True)
        st.markdown(f"**Why?** {row.get('top_reasons', '—')}")
        action_col = RISK_COLORS.get(risk, "#95a5a6")
        st.markdown(
            f'<div style="background:{action_col}22;border-left:4px solid {action_col};'
            f'padding:8px 14px;border-radius:6px;margin-top:6px;">'
            f'<b>MANAGER ACTION:</b> {row.get("manager_action","—")}</div>',
            unsafe_allow_html=True,
        )


def whatif_result_card(before: dict, after: dict):
    """Show before vs after for the What-If simulator."""
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Before")
        st.metric("Stock-out prob",  f"{before['prob']:.0%}")
        st.metric("Risk level",      before["risk"])
        st.metric("Recommended order", str(before["order"]))
        st.metric("Expected loss",   f"Rs {before['loss']:,.0f}")
    with cols[1]:
        st.subheader("After (What-If)")
        delta_prob = after["prob"] - before["prob"]
        st.metric("Stock-out prob",  f"{after['prob']:.0%}",
                  delta=f"{delta_prob:+.0%}", delta_color="inverse")
        st.metric("Risk level",      after["risk"])
        st.metric("Recommended order", str(after["order"]),
                  delta=str(after["order"] - before["order"]))
        st.metric("Expected loss",   f"Rs {after['loss']:,.0f}",
                  delta=f"Rs {after['loss']-before['loss']:+,.0f}", delta_color="inverse")
