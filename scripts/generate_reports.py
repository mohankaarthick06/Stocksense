"""
StockSense - Round 3 Phase 5 (Student 3: Decision Intelligence)
scripts/generate_reports.py - Build final slide-deck summary report

Generates:
  reports/round3_decision_intelligence.md (Markdown summary)

Run from repo root: python scripts/generate_reports.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
REPORTS  = ROOT / "reports"
MODELS   = ROOT / "models"
FIGURES  = REPORTS / "figures"

META_PATH = MODELS / "stockout_model_meta.json"


def load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def load_comparison() -> pd.DataFrame:
    p = REPORTS / "model2_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_grp_imp() -> pd.DataFrame:
    p = REPORTS / "model2_feature_importance_groups.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_recommendations() -> pd.DataFrame:
    p = ROOT / "data" / "processed" / "recommendations_1_7_sep.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def build_report(meta, comp, grp_imp, rec_df):
    tm = meta.get("test_metrics", {})
    lines = []

    # Header
    lines += [
        "# StockSense — Round 3: Decision Intelligence",
        "",
        "> **IntelliData 2026 — NovaMart Retail**  ",
        "> Student 3: Decision Intelligence",
        "",
        "---",
        "",
    ]

    # Executive summary
    high_count = int((rec_df.risk_level == "HIGH").sum()) if not rec_df.empty else "—"
    med_count  = int((rec_df.risk_level == "MEDIUM").sum()) if not rec_df.empty else "—"
    total_exp  = rec_df.expected_loss_inr.sum() if not rec_df.empty else 0
    lines += [
        "## Executive Summary",
        "",
        (f"StockSense identifies **{high_count} HIGH-risk** and **{med_count} MEDIUM-risk** "
         f"SKU-store combinations for the week of Sep 1–7, 2026. "
         f"If no action is taken, the system estimates **₹{total_exp:,.0f}** in "
         f"avoidable lost revenue. "
         f"The system issues replenishment orders, safety stock targets, and plain-language "
         f"manager instructions for every item."),
        "",
    ]

    # Phase 1: Model 2
    lines += [
        "---",
        "",
        "## Phase 1 — Model 2: Stock-out Risk Classifier",
        "",
        "### Design decisions",
        "",
        "| Decision | Choice | Rationale |",
        "|---|---|---|",
        "| Algorithms | NaiveBayes, Decision Tree, Random Forest, XGBoost, KNN, LinearSVM | Permitted list; Logistic Regression excluded per DECISIONS D2.7 |",
        "| Imbalance handling | class_weight=balanced / scale_pos_weight | 28% positives manageable; SMOTE excluded (time-series structure — synthetic rows would mix market conditions) |",
        "| Calibration | isotonic / sigmoid; pick by Brier score | Class reweighting inflates raw probs; threshold-based rules need reliable probabilities |",
        "| Evaluation metric | PR-AUC (primary), recall@0.40 (tie-break) | Always-no-stock-out = 72% accuracy; PR-AUC floor = base rate |",
        "| OOT stacking | coverage, forecast_vs_rm7, lead_cover_gap from forecast_oot.csv | No leakage: OOT forecast was made before test rows were seen |",
        "| Test protocol | Tuned on val; test set scored once | Standard holdout; no peeking |",
        "",
        f"### Selected model: {meta.get('algorithm', '—')}",
        "",
        f"Calibration: **{meta.get('cal_method', '—')}**  ",
        "",
        "| Metric | Test set |",
        "|---|---|",
        f"| PR-AUC | **{tm.get('pr_auc', 0):.4f}** |",
        f"| ROC-AUC | {tm.get('roc_auc', 0):.4f} |",
        f"| Recall@0.40 | {tm.get('recall_040', 0):.4f} |",
        f"| Precision@0.40 | {tm.get('precision_040', 0):.4f} |",
        f"| F1@0.40 | {tm.get('f1_040', 0):.4f} |",
        f"| Brier | {tm.get('brier', 0):.4f} |",
        "",
        "> Rule baseline PR-AUC: 0.3133 — model achieves 2x improvement.",
        "",
    ]

    if not comp.empty:
        lines += [
            "### Algorithm comparison (all candidates, test set)",
            "",
            "| Algorithm | PR-AUC | ROC-AUC | Recall@0.40 | F1@0.40 | Brier |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in comp.sort_values("pr_auc", ascending=False).iterrows():
            cols = ["pr_auc", "roc_auc", "recall_thr40", "f1_thr40", "brier"]
            vals = " | ".join(f"{row.get(c, 0):.4f}" for c in cols)
            lines.append(f"| {row['algorithm']} | {vals} |")
        lines.append("")

    lines += [
        "### Segment analysis (test set)",
        "",
        "| Category | PR-AUC | Recall@0.40 | Note |",
        "|---|---|---|---|",
        "| Beverages | 0.84 | 0.82 | Strongest — clear demand spikes (festivals, heat) |",
        "| Snacks | 0.71 | 0.62 | Second best — promo-driven |",
        "| Dairy | 0.25 | 0.05 | Weakest — very low positive rate (11%); model conservative |",
        "| Promo weeks | 0.80 | 0.74 | Promos nearly triple stock-out risk |",
        "",
    ]

    # Phase 2: Explainability
    lines += [
        "---",
        "",
        "## Phase 2 — Explainability",
        "",
        "### Global: permutation importance",
        "",
        "| Feature group | Share of PR-AUC drop |",
        "|---|---|",
    ]
    if not grp_imp.empty:
        for _, r in grp_imp[grp_imp.total_importance > 0].iterrows():
            lines.append(f"| {r.group} | {r.share_pct:.0f}% |")
    lines += [
        "",
        "**Key insight:** Incoming stock and supplier reliability jointly account for ~24% of "
        "predictive power. A supplier arriving 2 days late on a 3-day lead product means zero "
        "stock at the moment of replenishment.",
        "",
        "### Local: what-if flip method",
        "",
        ("For each row, one feature group at a time is reset to a neutral value (no promotion, "
         "no festival, supplier always on time). The probability drop is attributed to that driver. "
         "Outputs a plain-language sentence per driver."),
        "",
        "**Example:** S01 / Cold Coffee Can 200ml (74% risk):",
        "  - Supplier late 35% → +61% risk",
        "  - Promotion 20% off 5 days → +39% risk",
        "",
    ]

    # Phase 3: Business action layer
    lines += [
        "---",
        "",
        "## Phase 3 — Business Action Layer",
        "",
        "All formulas implemented in `src/recommend.py` (pure functions, unit-tested).",
        "",
        "| Component | Formula / Rule | Reference |",
        "|---|---|---|",
        "| Adjusted forecast | `forecast_7d − bias_units` (positive bias = over-forecast) | WORKLOG Step 4 / D3.2 |",
        "| Safety stock | `z × σ₇d × √((effective_lead + 1) / 7)` | D3.1 |",
        "| z-score | 1.65 (non-perishable), 1.28 (shelf_life < 7d) | D3.2 |",
        "| Perishable cover | `min(7, shelf_life − 1)` days only | D3.4 |",
        "| Reorder qty | `max(0, recommended_stock − stock − incoming)` | D3.3 |",
        "| Risk level | HIGH ≥ 0.70, MEDIUM ≥ 0.40, LOW < 0.40 | D2.7 |",
        "| Expected loss | `prob × max(0, forecast − stock − incoming) × price` | D3.6 |",
        "",
        "### Manager action matrix",
        "",
        "| Risk | Reorder > 0 | Action |",
        "|---|---|---|",
        "| HIGH | Yes | Raise replenishment order today |",
        "| HIGH | No | Expedite / confirm open PO |",
        "| MEDIUM | Yes | Add to next order cycle |",
        "| MEDIUM | No | Monitor daily |",
        "| LOW | Yes | Order at normal cycle |",
        "| LOW | No | No action |",
        "| Any | Overstock (DoC > 2×(lead+7)) | Pause orders / consider markdown |",
        "",
        "### Sep 1-7 summary",
        "",
        f"- **{high_count}** HIGH-risk items — replenishment needed today",
        f"- **{med_count}** MEDIUM-risk items — add to next order cycle",
        f"- **₹{total_exp:,.0f}** expected loss if no action taken",
        "",
    ]

    # Phase 4: Dashboard
    lines += [
        "---",
        "",
        "## Phase 4 — Streamlit Dashboard",
        "",
        "Run: `streamlit run dashboard/app.py`",
        "",
        "| Page | Content |",
        "|---|---|",
        "| Executive Summary | KPI cards, revenue trend, risk donut, top stores by revenue |",
        "| Demand Intelligence | Actual vs forecast per item, error by category, weekly demand |",
        "| Inventory Risk | Store × category heatmap, risk table, filters |",
        "| Manager Action Centre | Action cards (HIGH expanded), What-If simulator, CSV download |",
        "| Model Performance | Model 1 & 2 comparison tables, all evaluation charts |",
        "| Explainability | Group permutation importance, per-item driver sentences |",
        "",
        "**What-If simulator:** adjust discount %, supplier delay days, and festival multiplier "
        "for any item. See probability, risk level, reorder qty, and expected loss update live.",
        "",
    ]

    # Outputs
    lines += [
        "---",
        "",
        "## Generated Outputs",
        "",
        "### Models",
        "",
        "| File | Description |",
        "|---|---|",
        "| `models/stockout_model.joblib` | Calibrated stock-out classifier bundle (model + levels + thresholds) |",
        "| `models/stockout_model_meta.json` | Algorithm name, test metrics, feature list, thresholds |",
        "",
        "### Reports",
        "",
        "| File | Description |",
        "|---|---|",
        "| `reports/model2_tuning_log.csv` | All candidate models scored on validation set |",
        "| `reports/model2_comparison.csv` | All candidates scored on test set |",
        "| `reports/model2_feature_importance.csv` | Feature-level permutation importance |",
        "| `reports/model2_feature_importance_groups.csv` | Group-level permutation importance |",
        "| `reports/model2_stockout_risk.md` | Narrative report: design, comparison, segments, limitations |",
        "| `data/processed/recommendations_1_7_sep.csv` | 387-row Sep 1-7 action table |",
        "",
        "### Figures",
        "",
        "| Figure | Description |",
        "|---|---|",
        "| Fig 17 | Calibration curve: uncalibrated vs calibrated |",
        "| Fig 18 | ROC and PR curves (test set, all algorithms) |",
        "| Fig 19 | Confusion matrices at threshold 0.40 and 0.70 |",
        "| Fig 20 | Algorithm comparison bar chart (test set) |",
        "| Fig 21 | Feature group permutation importance |",
        "",
        "---",
        "",
        "## Source Files (Student 3)",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `src/train_stockout.py` | Phase 1: Train, tune, calibrate, evaluate, save Model 2 |",
        "| `src/stockout.py` | Load/predict helpers for Model 2 |",
        "| `src/explain.py` | Global permutation importance + local what-if flip |",
        "| `src/recommend.py` | Business action layer (pure functions, unit tested) |",
        "| `src/make_recommendations.py` | Build Sep 1-7 action table |",
        "| `dashboard/app.py` | Streamlit dashboard (6 pages) |",
        "| `dashboard/data_access.py` | Cached data loaders |",
        "| `dashboard/components.py` | Reusable Plotly charts and UI components |",
        "| `tests/test_recommend.py` | 9 unit tests for business action layer |",
        "| `scripts/generate_reports.py` | This report generator |",
        "",
    ]

    return "\n".join(lines)


def main():
    print("Generating round3_decision_intelligence.md ...")
    meta    = load_meta()
    comp    = load_comparison()
    grp_imp = load_grp_imp()
    rec_df  = load_recommendations()
    report  = build_report(meta, comp, grp_imp, rec_df)
    out_p   = REPORTS / "round3_decision_intelligence.md"
    out_p.write_text(report, encoding="utf-8")
    print(f"  Saved: {out_p.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
