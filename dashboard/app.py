"""
StockSense — Streamlit Dashboard (Student 3: Decision Intelligence)
dashboard/app.py

Six pages:
  1. Executive Summary
  2. Demand Intelligence
  3. Inventory Risk
  4. Manager Action Centre  (+ What-If sidebar)
  5. Model Performance
  6. Explainability

Run from repo root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import data_access as da
import components  as co

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense | IntelliData 2026",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    # Streamlit light theme is set via .streamlit/config.toml
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base canvas ────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}
.stApp, .main, section.main {
    background-color: #FFFFFF !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 2px 0 12px rgba(37,99,235,0.06);
}
[data-testid="stSidebar"] * { color: #1B2A4A !important; }
[data-testid="stSidebar"] .stRadio label { font-weight: 500; font-size: 0.95rem; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    border-radius: 8px;
    padding: 4px 6px;
}

/* ── Metric cards ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #1B2A4A !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Section headers ────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1B2A4A;
    border-bottom: 3px solid #2563EB;
    padding-bottom: 6px;
    margin-bottom: 16px;
    letter-spacing: -0.01em;
}

/* ── Expanders ──────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 6px;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #1B2A4A;
}

/* ── Info / warning boxes ───────────────────────────────────────── */
.stAlert {
    border-radius: 10px;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
    color: #64748B;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 3px solid #2563EB;
}

/* ── Dataframe ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}

/* ── Buttons ────────────────────────────────────────────────────── */
[data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: #FFFFFF;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    transition: all 0.2s;
}
[data-testid="stButton"] > button:hover, [data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(37,99,235,0.4);
}

/* ── Selectbox / sliders ────────────────────────────────────────── */
[data-baseweb="select"] {
    border-radius: 8px !important;
    background: #FFFFFF;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #2563EB;
}

/* ── Page title ─────────────────────────────────────────────────── */
h1 { color: #1B2A4A !important; font-weight: 800 !important; letter-spacing: -0.02em; }
h2, h3 { color: #1B2A4A !important; font-weight: 700 !important; }
p, li, .stMarkdown { color: #374151 !important; }

/* ── Caption ────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #94A3B8 !important; font-size: 0.8rem; }

/* ── Big number helper ──────────────────────────────────────────── */
.big-number { font-size: 2.2rem; font-weight: 800; color: #2563EB; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/64/000000/box.png", width=48)
    st.title("StockSense")
    st.caption("IntelliData 2026 — NovaMart Retail")
    st.markdown("---")

    PAGE = st.radio(
        "Navigate",
        [
            "📊 Executive Summary",
            "📈 Demand Intelligence",
            "⚠️ Inventory Risk",
            "🎯 Manager Action Centre",
            "🤖 Model Performance",
            "🔍 Explainability",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Global filters**")
    rec_df_all = da.load_recommendations()
    stores_all     = sorted(rec_df_all.store.unique())    if not rec_df_all.empty else []
    categories_all = sorted(rec_df_all.category.unique()) if not rec_df_all.empty else []
    risk_opts      = ["HIGH", "MEDIUM", "LOW"]

    sel_stores = st.multiselect("Store", stores_all, default=stores_all)
    sel_cats   = st.multiselect("Category", categories_all, default=categories_all)
    sel_risks  = st.multiselect("Risk level", risk_opts, default=risk_opts)

# Apply global filter to recommendations
if not rec_df_all.empty:
    rec_df = rec_df_all[
        rec_df_all.store.isin(sel_stores) &
        rec_df_all.category.isin(sel_cats) &
        rec_df_all.risk_level.isin(sel_risks)
    ].copy()
else:
    rec_df = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Executive Summary
# ═══════════════════════════════════════════════════════════════════════════
if PAGE == "📊 Executive Summary":
    st.title("📊 Executive Summary")
    st.caption("NovaMart Retail — StockSense decision intelligence system")

    master = da.load_master()
    kpi    = da.load_kpi_summary()

    # --- KPI row ---------
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    with k1:
        revenue = master.revenue.sum()
        st.metric("Total Revenue", f"Rs {revenue/1e7:.2f} Cr")
    with k2:
        units = master.units_sold.sum()
        st.metric("Units Sold", f"{units/1e6:.2f} M")
    with k3:
        so_rate = master.stockout_flag.mean() if "stockout_flag" in master.columns else 0
        st.metric("Stock-out Rate", f"{so_rate:.1%}")
    with k4:
        inv_val = (master.closing_stock * master.unit_price).sum() if "unit_price" in master.columns else 0
        st.metric("Inventory Value", f"Rs {inv_val/1e7:.1f} Cr")
    with k5:
        lost = master.lost_sales_est.sum() if "lost_sales_est" in master.columns else 0
        st.metric("Est. Lost Sales", f"Rs {lost/1e5:.1f} L")
    with k6:
        high_count = (rec_df_all.risk_level == "HIGH").sum() if not rec_df_all.empty else 0
        st.metric("HIGH Risk Items (Sep)", str(high_count), delta="Sep 1-7 forecast")
    with k7:
        total_exp_loss = rec_df_all.expected_loss_inr.sum() if not rec_df_all.empty else 0
        st.metric("Expected Loss at Risk", f"Rs {total_exp_loss/1e5:.1f} L")

    st.markdown("---")

    # --- Headline action ---
    if not rec_df_all.empty:
        top1 = rec_df_all.iloc[0]
        st.info(
            f"**⚡ Top action today:** {top1['store']} — {top1['product']} | "
            f"Risk: {top1['risk_level']} | Order {top1['recommended_order']} units | "
            f"Expected loss if no action: Rs {top1['expected_loss_inr']:,.0f}"
        )

    # --- Charts row ---
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="section-header">Revenue trend</div>', unsafe_allow_html=True)
        st.plotly_chart(co.revenue_trend(master), use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Risk distribution (Sep 1-7)</div>', unsafe_allow_html=True)
        if not rec_df_all.empty:
            st.plotly_chart(co.risk_donut(rec_df_all), use_container_width=True)

    # --- Top stores by revenue ---
    st.markdown('<div class="section-header">Revenue by store</div>', unsafe_allow_html=True)
    store_rev = master.groupby("store_id").revenue.sum().sort_values(ascending=False).reset_index()
    fig = px.bar(store_rev, x="store_id", y="revenue",
                 labels={"store_id": "Store", "revenue": "Revenue (Rs)"},
                 color="revenue", color_continuous_scale=["#2980b9", "#e74c3c"],
                 title="Total revenue by store (Mar-Aug)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Top categories ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Revenue by category</div>', unsafe_allow_html=True)
        cat_rev = (master.groupby("category").revenue.sum()
                   .sort_values(ascending=False).reset_index())
        fig2 = px.bar(cat_rev, x="revenue", y="category", orientation="h",
                      color="revenue", color_continuous_scale=["#2980b9", "#e74c3c"])
        fig2.update_layout(height=320, margin=dict(t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Stock-out rate by category</div>', unsafe_allow_html=True)
        so_cat = (master.groupby("category").stockout_flag.mean()
                  .sort_values(ascending=False).reset_index())
        fig3 = px.bar(so_cat, x="stockout_flag", y="category", orientation="h",
                      color="stockout_flag", color_continuous_scale=["#2ecc71", "#e74c3c"],
                      labels={"stockout_flag": "Stock-out rate"})
        fig3.update_layout(height=320, margin=dict(t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Demand Intelligence
# ═══════════════════════════════════════════════════════════════════════════
elif PAGE == "📈 Demand Intelligence":
    st.title("📈 Demand Intelligence")

    oot   = da.load_forecast_oot()
    feat  = da.load_features()
    err   = da.load_error_by_category()

    # Store x product selector
    combos = feat[["store_id", "product_id", "product_name"]].drop_duplicates()
    store_opts = sorted(combos.store_id.unique())
    sel_store = st.selectbox("Store", store_opts)
    prod_opts = sorted(combos[combos.store_id == sel_store].product_name.unique())
    sel_prod_name = st.selectbox("Product", prod_opts)
    prod_id = combos[(combos.store_id == sel_store) &
                     (combos.product_name == sel_prod_name)].product_id.iloc[0]

    st.plotly_chart(
        co.actual_vs_forecast_chart(oot, feat, sel_store, prod_id),
        use_container_width=True,
    )

    # Sep forecast for this item
    fnxt = da.load_forecast_next7()
    item_fcast = fnxt[(fnxt.store_id == sel_store) & (fnxt.product_id == prod_id)]
    if not item_fcast.empty:
        r = item_fcast.iloc[0]
        st.info(
            f"**Sep 1-7 forecast:** {r.forecast_7d:.0f} units "
            f"(P10={r.forecast_p10:.0f} / P90={r.forecast_p90:.0f}) | "
            f"Sigma: {r.forecast_sigma_7d:.1f}"
        )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Forecast error by category</div>', unsafe_allow_html=True)
        st.plotly_chart(co.wape_by_category(err), use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Demand by category (weekly)</div>', unsafe_allow_html=True)
        master = da.load_master()
        weekly_cat = (
            master.assign(week=master.date.dt.to_period("W").dt.start_time)
            .groupby(["week", "category"]).demand_est.sum().reset_index()
        )
        fig = px.line(weekly_cat, x="week", y="demand_est", color="category",
                      labels={"week": "Week", "demand_est": "Demand (units)"},
                      color_discrete_sequence=co.PALETTE)
        fig.update_layout(height=360, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Model 1 metrics table
    st.markdown('<div class="section-header">Model 1 metrics (test set)</div>', unsafe_allow_html=True)
    m1 = da.load_model1_comparison()
    st.dataframe(m1.style.highlight_min(subset=["WAPE"], color="#2ecc71")
                         .highlight_max(subset=["WAPE"], color="#e74c3c"),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Inventory Risk
# ═══════════════════════════════════════════════════════════════════════════
elif PAGE == "⚠️ Inventory Risk":
    st.title("⚠️ Inventory Risk (Sep 1-7 forecast)")

    if rec_df.empty:
        st.warning("No recommendations loaded. Run src/make_recommendations.py first.")
        st.stop()

    # Risk counts KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 HIGH",   str((rec_df.risk_level == "HIGH").sum()))
    k2.metric("🟡 MEDIUM", str((rec_df.risk_level == "MEDIUM").sum()))
    k3.metric("🟢 LOW",    str((rec_df.risk_level == "LOW").sum()))
    k4.metric("Total items", str(len(rec_df)))

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="section-header">Store x Category heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(co.stock_heatmap(rec_df), use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Risk distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(co.risk_donut(rec_df), use_container_width=True)

    # Sortable risk table
    st.markdown('<div class="section-header">Risk detail table</div>', unsafe_allow_html=True)

    show_cols = [
        "store", "product", "category", "risk_level",
        "current_stock", "incoming_stock", "forecast_7d",
        "stockout_probability", "recommended_order", "expected_loss_inr",
    ]
    st.dataframe(
        rec_df[show_cols].sort_values("expected_loss_inr", ascending=False)
        .style.background_gradient(subset=["stockout_probability"], cmap="RdYlGn_r")
              .background_gradient(subset=["expected_loss_inr"], cmap="Reds"),
        use_container_width=True,
        height=500,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Manager Action Centre
# ═══════════════════════════════════════════════════════════════════════════
elif PAGE == "🎯 Manager Action Centre":
    st.title("🎯 Manager Action Centre")

    if rec_df.empty:
        st.warning("No recommendations found. Run src/make_recommendations.py first.")
        st.stop()

    # What-If sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 What-If Simulator")
        wi_stores = sorted(rec_df["store"].unique())
        wi_store  = st.selectbox("What-If store", wi_stores, key="wi_store")
        wi_prods  = sorted(rec_df[rec_df["store"] == wi_store]["product"].unique())
        wi_prod   = st.selectbox("What-If product", wi_prods, key="wi_prod")
        st.markdown("**Adjust scenario:**")
        wi_discount    = st.slider("Additional discount (%)", 0, 50, 0, key="wi_disc")
        wi_delay       = st.slider("Supplier delay (+days)", 0, 10, 0, key="wi_delay")
        wi_fest_mult   = st.slider("Festival demand multiplier (%)", 0, 50, 0, key="wi_fest")
        run_whatif     = st.button("▶ Run What-If", key="run_wi")

    # Top 10 today
    st.markdown('<div class="section-header">⚡ Top 10 actions today (by expected loss)</div>',
                unsafe_allow_html=True)
    top10 = rec_df.head(10)
    for _, row in top10.iterrows():
        co.recommendation_card(row)

    st.markdown("---")

    # Download button
    st.download_button(
        "📥 Download full action list (CSV)",
        data=rec_df.to_csv(index=False).encode("utf-8"),
        file_name="stocksense_actions_sep2026.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # All cards
    st.markdown('<div class="section-header">All recommendations</div>', unsafe_allow_html=True)
    for _, row in rec_df.iterrows():
        co.recommendation_card(row)

    # What-If result
    if run_whatif:
        st.markdown("---")
        st.markdown("### What-If result")
        wi_row_orig = rec_df[(rec_df["store"] == wi_store) & (rec_df["product"] == wi_prod)]
        if wi_row_orig.empty:
            st.warning("Selected item not in current filter.")
        else:
            # Load the full row from features / forecast
            feat_full = da.load_features()
            fnxt_full = da.load_forecast_next7()

            from recommend import build_recommendation, get_sigma_7d, reorder_qty, risk_level as rl_fn
            from stockout import predict_proba as so_predict, load_model as so_load
            bundle = da.load_stockout_model()

            orig_row = wi_row_orig.iloc[0]
            pid = orig_row.get("product", "")

            # Find matching features row
            feat_row_df = feat_full[
                (feat_full["store_id"] == wi_store) &
                (feat_full["product_name"] == wi_prod) &
                (feat_full.date == pd.Timestamp("2026-08-31"))
            ]
            if feat_row_df.empty:
                st.warning("Could not find feature row for What-If.")
            else:
                feat_row = feat_row_df.iloc[0].copy()
                fnxt_row = fnxt_full[(fnxt_full["store_id"] == wi_store) &
                                     (fnxt_full["product_name"] == wi_prod)]
                if not fnxt_row.empty:
                    feat_row["forecast_7d"]       = float(fnxt_row.iloc[0].forecast_7d)
                    feat_row["forecast_sigma_7d"]  = float(fnxt_row.iloc[0].forecast_sigma_7d)

                # Stacked features for base row
                feat_row["coverage"]        = ((feat_row.closing_stock + feat_row.incoming_next7)
                                                / max(feat_row.forecast_7d, 1))
                feat_row["forecast_vs_rm7"] = (feat_row.forecast_7d
                                                / max(feat_row.rolling_mean_7 * 7, 1))
                feat_row["lead_cover_gap"]  = (feat_row.closing_stock + feat_row.incoming_next7
                                                - feat_row.forecast_7d
                                                  * (feat_row.effective_lead_time / 7))

                before_prob = float(so_predict(bundle, pd.DataFrame([feat_row]))[0])
                before_rec  = build_recommendation(feat_row, before_prob)
                before_dict = {"prob": before_prob, "risk": before_rec["risk_level"],
                               "order": before_rec["recommended_order"],
                               "loss": before_rec["expected_loss_inr"]}

                # Apply What-If changes
                wi_feat = feat_row.copy()
                if wi_discount > 0:
                    wi_feat["promo_days_next7"]  = min(7, wi_feat.get("promo_days_next7", 0) + 3)
                    wi_feat["max_discount_next7"] = max(wi_feat.get("max_discount_next7", 0),
                                                        wi_discount / 100)
                    wi_feat["avg_discount_next7"] = max(wi_feat.get("avg_discount_next7", 0),
                                                        wi_discount / 100 * 0.7)
                    wi_feat["forecast_7d"]        = wi_feat["forecast_7d"] * (1 + wi_discount * 0.01)

                if wi_delay > 0:
                    wi_feat["effective_lead_time"] = wi_feat.get("effective_lead_time", 3) + wi_delay
                    wi_feat["supplier_avg_delay"]   = wi_feat.get("supplier_avg_delay", 0) + wi_delay
                    # Subtract incoming that would now arrive after the window
                    wi_feat["incoming_next7"] = max(0, wi_feat.get("incoming_next7", 0)
                                                    - wi_feat.get("incoming_next7", 0) * (wi_delay / 7))

                if wi_fest_mult > 0:
                    wi_feat["forecast_7d"] = wi_feat["forecast_7d"] * (1 + wi_fest_mult / 100)

                # Recompute stacked features
                wi_feat["coverage"]       = ((wi_feat.get("closing_stock", 0)
                                               + wi_feat.get("incoming_next7", 0))
                                              / max(wi_feat.get("forecast_7d", 1), 1))
                wi_feat["forecast_vs_rm7"] = (wi_feat.get("forecast_7d", 0)
                                               / max(wi_feat.get("rolling_mean_7", 1) * 7, 1))
                wi_feat["lead_cover_gap"]  = (wi_feat.get("closing_stock", 0)
                                               + wi_feat.get("incoming_next7", 0)
                                               - wi_feat.get("forecast_7d", 0)
                                                 * (wi_feat.get("effective_lead_time", 3) / 7))

                after_prob = float(so_predict(bundle, pd.DataFrame([wi_feat]))[0])
                after_rec  = build_recommendation(wi_feat, after_prob)
                after_dict = {"prob": after_prob, "risk": after_rec["risk_level"],
                              "order": after_rec["recommended_order"],
                              "loss": after_rec["expected_loss_inr"]}

                co.whatif_result_card(before_dict, after_dict)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: Model Performance
# ═══════════════════════════════════════════════════════════════════════════
elif PAGE == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    tab1, tab2 = st.tabs(["Model 1 — Demand Forecast", "Model 2 — Stock-out Risk"])

    with tab1:
        st.markdown("### Algorithm comparison (test set)")
        m1 = da.load_model1_comparison()
        st.dataframe(m1, use_container_width=True)
        st.caption("Selected: XGBoost (lowest WAPE 15.3%, beats baseline 20.8%)")

        fig_paths = {
            "Actual vs Forecast": da.figure_path(12),
            "Residual analysis":  da.figure_path(14),
            "Feature importance": da.figure_path(15),
            "Model comparison":   da.figure_path(11),
        }
        cols = st.columns(2)
        for i, (title, p) in enumerate(fig_paths.items()):
            if p and p.exists():
                with cols[i % 2]:
                    st.markdown(f"**{title}**")
                    st.image(str(p), use_container_width=True)

    with tab2:
        st.markdown("### Algorithm comparison (test set)")
        m2 = da.load_model2_comparison()
        if not m2.empty:
            st.dataframe(
                m2[["algorithm", "pr_auc", "roc_auc", "recall_thr40",
                     "f1_thr40", "brier"]].sort_values("pr_auc", ascending=False),
                use_container_width=True,
            )
            st.caption(
                "Selected: calibrated LinearSVM (highest val PR-AUC 0.6134). "
                "Test PR-AUC 0.6299 vs rule baseline 0.31 — 2x improvement."
            )

        st.markdown("### Why PR-AUC and recall, not accuracy")
        st.info(
            "A classifier that always predicts 'no stock-out' achieves ~72% accuracy "
            "while missing every real event. PR-AUC has a floor equal to the base rate (0.28); "
            "it cannot be fooled by predicting the majority class. "
            "Recall is primary: a missed stock-out = lost revenue. "
            "A false alarm = a phone call."
        )

        fig_paths_2 = {
            "Calibration curve (Fig 17)":   da.figure_path(17),
            "ROC & PR curves (Fig 18)":     da.figure_path(18),
            "Confusion matrices (Fig 19)":  da.figure_path(19),
            "Model comparison (Fig 20)":    da.figure_path(20),
        }
        cols2 = st.columns(2)
        for i, (title, p) in enumerate(fig_paths_2.items()):
            if p and p.exists():
                with cols2[i % 2]:
                    st.markdown(f"**{title}**")
                    st.image(str(p), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: Explainability
# ═══════════════════════════════════════════════════════════════════════════
elif PAGE == "🔍 Explainability":
    st.title("🔍 Explainability")

    tab1, tab2 = st.tabs(["Global importance", "Per-item explainer"])

    with tab1:
        st.markdown("### What drives stock-out risk? (global view)")
        st.caption(
            "Permutation importance on the test set (Aug 11-24). "
            "Each bar = mean PR-AUC drop when that feature group is randomly shuffled. "
            "Larger drop = more important."
        )
        grp_imp = da.load_feature_importance()
        if not grp_imp.empty:
            st.plotly_chart(co.feature_importance_bar(grp_imp), use_container_width=True)

        p21 = da.figure_path(21)
        if p21 and p21.exists():
            st.image(str(p21), use_container_width=True)

        st.markdown("### Feature group definitions")
        groups_info = {
            "Recent sales trend":
                "Level and momentum of past demand (rolling means, lags). "
                "Shows whether demand is rising or falling into the forecast window.",
            "Incoming stock & supplier reliability":
                "Open purchase orders expected within 7 days + supplier's historical late rate. "
                "A late supplier effectively extends lead time and increases risk.",
            "Current stock & cover":
                "Closing stock, reorder gap, coverage ratio (stock / forecast). "
                "The most direct signal: if coverage < 1, a stock-out is likely.",
            "Store & product profile":
                "Store size, format, avg customers. Larger stores have higher absolute demand "
                "and different ordering patterns.",
            "Festival / holiday / payday calendar":
                "Festivals drive pre-festival buying (pooja items, ghee, sweets). "
                "Payday weeks increase demand for staples.",
            "Promotion plan":
                "Planned discount and promo days in the next 7 days. "
                "Promotions roughly double stock-out risk.",
            "Stock-out history": "Past stock-out days — chronic problems repeat.",
            "Price": "Price changes affect demand, especially after MRP hikes.",
            "Weather": "Temperature drives beverages and ice cream sales.",
        }
        for grp, desc in groups_info.items():
            with st.expander(grp):
                st.write(desc)

    with tab2:
        st.markdown("### Per-item explanation (what-if flip method)")
        st.caption(
            "For each item, one feature group at a time is reset to a neutral value "
            "(no promotion, no festival, supplier always on time, etc.) and the model "
            "is re-run. The probability drop is attributed to that driver."
        )

        if rec_df.empty:
            st.warning("No recommendations loaded.")
            st.stop()

        stores_e = sorted(rec_df["store"].unique())
        sel_s_e  = st.selectbox("Store", stores_e, key="exp_store")
        prods_e  = sorted(rec_df[rec_df["store"] == sel_s_e]["product"].unique())
        sel_p_e  = st.selectbox("Product", prods_e, key="exp_prod")

        item_row = rec_df[(rec_df["store"] == sel_s_e) & (rec_df["product"] == sel_p_e)]
        if not item_row.empty:
            r = item_row.iloc[0]
            prob = float(r.get("stockout_probability", 0))
            risk = str(r.get("risk_level", "LOW"))
            reasons = str(r.get("top_reasons", ""))

            st.markdown(
                f"**Stock-out probability:** {prob:.0%}  |  "
                f"**Risk:** {co.risk_badge(risk)}",
                unsafe_allow_html=True,
            )

            if reasons:
                st.markdown("**Top drivers (what-if flip attribution):**")
                for part in reasons.split(" | "):
                    st.markdown(f"- {part}")

            st.markdown("**Manager action:** " + str(r.get("manager_action", "—")))

            # Try to get the full explanation with driver sentences
            feat_full = da.load_features()
            bundle    = da.load_stockout_model()
            fnxt_full = da.load_forecast_next7()

            feat_rows = feat_full[
                (feat_full.store_id == sel_s_e) &
                (feat_full.product_name == sel_p_e) &
                (feat_full.date == pd.Timestamp("2026-08-31"))
            ]
            if not feat_rows.empty:
                feat_row = feat_rows.iloc[0].copy()
                fnxt_row = fnxt_full[(fnxt_full.store_id == sel_s_e) &
                                     (fnxt_full.product_name == sel_p_e)]
                if not fnxt_row.empty:
                    feat_row["forecast_7d"] = float(fnxt_row.iloc[0].forecast_7d)

                feat_row["coverage"]        = ((feat_row.closing_stock + feat_row.incoming_next7)
                                                / max(feat_row.forecast_7d, 1))
                feat_row["forecast_vs_rm7"] = (feat_row.forecast_7d
                                                / max(feat_row.rolling_mean_7 * 7, 1))
                feat_row["lead_cover_gap"]  = (feat_row.closing_stock + feat_row.incoming_next7
                                                - feat_row.forecast_7d
                                                  * (feat_row.effective_lead_time / 7))

                try:
                    from explain import explain_row, FEATURE_GROUPS_LABELS
                    feat_medians = {
                        f: feat_full[f].median()
                        for f in bundle["features"]
                        if f in feat_full.columns
                        and pd.api.types.is_numeric_dtype(feat_full[f])
                    }
                    drivers = explain_row(feat_row, bundle, feat_medians=feat_medians)
                    if drivers:
                        st.markdown("**Full driver breakdown:**")
                        for grp, pct, sent in drivers:
                            icon = "🔴" if pct > 40 else "🟡" if pct > 20 else "🟢"
                            st.markdown(f"{icon} **{pct:.0f}%** — {sent}")
                except Exception as e:
                    st.caption(f"(Driver detail unavailable: {e})")
