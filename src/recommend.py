"""
StockSense - Round 3 Phase 3 (Student 3: Decision Intelligence)
src/recommend.py - Business action layer (pure functions, easy to unit test and reuse in What-If)

Implements DECISIONS D3.1 - D3.6:
  safety_stock, reorder_qty, risk_level, manager_action, expected_loss_inr

Bias sign note (verified against data):
  forecast_error_by_combo.csv bias_units = mean(forecast - actual).
  Positive value = model over-forecasts that item.
  adjusted_forecast = forecast_7d - bias_units
  (subtract the over-forecast to get an unbiased demand estimate).
  This is documented in WORKLOG Step 4 (Person 2).

Run: python src/recommend.py  (prints a self-test)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS   = ROOT / "reports"

# Service level z-scores (D3.1)
Z_NORMAL     = 1.65   # 95% service level
Z_PERISHABLE = 1.28   # lower z for perishables to reduce waste
PERISHABLE_SHELF_THRESHOLD = 7   # shelf_life_days < 7 -> perishable logic


# ---------------------------------------------------------------------------
# Load error profiles (cached once)
# ---------------------------------------------------------------------------
_combo_err: Optional[pd.DataFrame] = None
_cat_err:   Optional[pd.DataFrame] = None


def _load_error_profiles():
    global _combo_err, _cat_err
    if _combo_err is None:
        _combo_err = pd.read_csv(PROCESSED / "forecast_error_by_combo.csv")
        _cat_err   = pd.read_csv(REPORTS   / "forecast_error_by_category.csv")


def get_sigma_7d(store_id: str, product_id: str, category: str,
                 fallback_sigma: Optional[float] = None) -> float:
    """
    Item-level sigma_7d from forecast_error_by_combo.csv.
    Falls back to category-level from forecast_error_by_category.csv
    when the item has fewer than 28 rows of history (WORKLOG Step 5, Person 2).
    """
    _load_error_profiles()
    row = _combo_err[(_combo_err.store_id == store_id) &
                     (_combo_err.product_id == product_id)]
    if len(row) > 0 and row.iloc[0]["rows"] >= 28:
        return float(row.iloc[0]["sigma_7d"])

    cat_row = _cat_err[_cat_err.category_name == category]
    if len(cat_row) > 0:
        return float(cat_row.iloc[0]["sigma_7d"])

    return fallback_sigma if fallback_sigma is not None else 20.0


def get_bias_units(store_id: str, product_id: str) -> float:
    """
    bias_units = mean(forecast - actual) for this item.
    Positive = model over-forecasts -> subtract to get adjusted_forecast.
    Zero when item not found (no adjustment).
    """
    _load_error_profiles()
    row = _combo_err[(_combo_err.store_id == store_id) &
                     (_combo_err.product_id == product_id)]
    if len(row) > 0:
        return float(row.iloc[0]["bias_units"])
    return 0.0


# ---------------------------------------------------------------------------
# Safety stock (D3.1 / D3.2)
# ---------------------------------------------------------------------------
def safety_stock(sigma_7d: float, effective_lead: float,
                 shelf_life_days: float, review_days: int = 1) -> float:
    """
    safety_stock = z * sigma_7d * sqrt((effective_lead + review_days) / 7)

    z = 1.28 for perishables (shelf_life_days < 7), else 1.65.
    Effective lead already includes supplier average delay (D3.2).
    """
    z = Z_PERISHABLE if shelf_life_days < PERISHABLE_SHELF_THRESHOLD else Z_NORMAL
    factor = max((effective_lead + review_days) / 7, 0.0)
    return z * sigma_7d * math.sqrt(factor)


# ---------------------------------------------------------------------------
# Perishable cover cap (D3.4)
# ---------------------------------------------------------------------------
def perishable_cover_days(shelf_life_days: float) -> float:
    """
    For perishables (shelf_life < 7): cover only min(7, shelf_life - 1) days.
    This prevents ordering a full 7-day supply of 3-day milk.
    """
    if shelf_life_days < PERISHABLE_SHELF_THRESHOLD:
        return min(7.0, max(shelf_life_days - 1.0, 1.0))
    return 7.0


def adjusted_forecast_cover(forecast_7d: float, shelf_life_days: float) -> float:
    """Scale the 7-day forecast down for perishables (D3.4)."""
    cover_days = perishable_cover_days(shelf_life_days)
    return forecast_7d * (cover_days / 7.0)


# ---------------------------------------------------------------------------
# Reorder quantity (D3.3)
# ---------------------------------------------------------------------------
def reorder_qty(
    forecast_7d:        float,
    sigma_7d:           float,
    effective_lead:     float,
    closing_stock:      float,
    incoming_next7:     float,
    shelf_life_days:    float,
    bias_units:         float = 0.0,
    review_days:        int   = 1,
) -> dict:
    """
    Compute recommended_stock and reorder_qty.

    adjusted_forecast = forecast_7d - bias_units
      (bias = mean(forecast - actual); positive bias -> over-forecast -> subtract)

    recommended_stock = adjusted_forecast_cover + safety_stock
    reorder = max(0, recommended_stock - current_stock - incoming)
    Rounded up to integer (no case-pack info; stated in comments).

    Returns a dict with all intermediate values for transparency.
    """
    adj_forecast = max(forecast_7d - bias_units, 0.0)
    forecast_cover = adjusted_forecast_cover(adj_forecast, shelf_life_days)
    ss = safety_stock(sigma_7d, effective_lead, shelf_life_days, review_days)
    rec_stock = forecast_cover + ss
    raw_order = rec_stock - closing_stock - incoming_next7
    order = math.ceil(max(0.0, raw_order))   # no case-pack column -> unit rounding

    return {
        "adj_forecast":    round(adj_forecast, 1),
        "forecast_cover":  round(forecast_cover, 1),
        "safety_stock":    round(ss, 1),
        "recommended_stock": round(rec_stock, 1),
        "reorder_qty":     order,
    }


# ---------------------------------------------------------------------------
# Risk level (D2.7)
# ---------------------------------------------------------------------------
def risk_level(prob: float, thr_high: float = 0.70, thr_medium: float = 0.40) -> str:
    if prob >= thr_high:
        return "HIGH"
    if prob >= thr_medium:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Manager action (D3.5)
# ---------------------------------------------------------------------------
def manager_action(
    risk:           str,
    order:          int,
    days_of_cover:  float,
    effective_lead: float,
) -> str:
    """
    Decision matrix per DECISIONS D3.5:
    Risk x (reorder > 0) -> unambiguous instruction.
    Overstock check: days_of_cover > 2 * (effective_lead + 7).
    """
    overstock_threshold = 2.0 * (effective_lead + 7.0)
    if days_of_cover > overstock_threshold:
        return "Overstock: pause orders / consider markdown"

    if risk == "HIGH":
        if order > 0:
            return "Raise replenishment order today"
        else:
            return "Expedite / confirm open PO"

    if risk == "MEDIUM":
        if order > 0:
            return "Add to next order cycle"
        else:
            return "Monitor daily"

    # LOW
    if order > 0:
        return "Order at normal cycle"
    return "No action"


# ---------------------------------------------------------------------------
# Expected loss (D3.6)
# ---------------------------------------------------------------------------
def expected_loss_inr(
    stockout_prob:  float,
    forecast_7d:    float,
    closing_stock:  float,
    incoming_next7: float,
    unit_price:     float,
    cost_price:     float = 0.0,
) -> dict:
    """
    expected_shortfall_units = max(0, forecast_7d - closing_stock - incoming_next7)
    expected_loss_inr = stockout_prob * expected_shortfall_units * unit_price
    margin_at_risk    = stockout_prob * expected_shortfall_units * (unit_price - cost_price)
    """
    shortfall = max(0.0, forecast_7d - closing_stock - incoming_next7)
    loss = stockout_prob * shortfall * unit_price
    margin = stockout_prob * shortfall * max(unit_price - cost_price, 0.0)
    return {
        "expected_shortfall_units": round(shortfall, 1),
        "expected_loss_inr":        round(loss, 2),
        "margin_at_risk_inr":       round(margin, 2),
    }


# ---------------------------------------------------------------------------
# Full recommendation for one row
# ---------------------------------------------------------------------------
def build_recommendation(
    row: pd.Series,
    stockout_prob: float,
    top_reasons_str: str = "",
    review_days: int = 1,
) -> dict:
    """
    Given a row from forecast_next7.csv (or a features.csv row augmented with
    forecast_7d / forecast_sigma_7d) and its calibrated stock-out probability,
    return the full recommendation dict.

    Columns expected in `row`:
      store_id, product_id, product_name, category, store_type,
      unit_price, cost_price (optional), closing_stock, incoming_next7,
      forecast_7d, forecast_sigma_7d, effective_lead_time (or lead_time_days),
      shelf_life_days, days_of_cover,
      forecast_p10 (optional), forecast_p90 (optional).
    """
    def g(col, default=0.0):
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    store_id   = str(g("store_id",   "?"))
    product_id = str(g("product_id", "?"))
    category   = str(g("category",   g("category_name", "?")))

    sigma_7d       = get_sigma_7d(store_id, product_id, category,
                                  fallback_sigma=float(g("forecast_sigma_7d", 20.0)))
    bias           = get_bias_units(store_id, product_id)
    effective_lead = float(g("effective_lead_time", g("lead_time_days", 3.0)))
    shelf_life     = float(g("shelf_life_days", 30.0))
    forecast_7d    = float(g("forecast_7d", 0.0))
    closing_stock  = float(g("closing_stock", 0.0))
    incoming       = float(g("incoming_next7", 0.0))
    unit_price     = float(g("unit_price", 0.0))
    cost_price     = float(g("cost_price", 0.0))
    doc            = float(g("days_of_cover", 0.0))

    rq = reorder_qty(forecast_7d, sigma_7d, effective_lead,
                     closing_stock, incoming, shelf_life, bias, review_days)
    rl = risk_level(stockout_prob)
    action = manager_action(rl, rq["reorder_qty"], doc, effective_lead)
    loss = expected_loss_inr(stockout_prob, forecast_7d, closing_stock, incoming,
                              unit_price, cost_price)

    p10 = g("forecast_p10", forecast_7d * 0.8)
    p90 = g("forecast_p90", forecast_7d * 1.2)

    return {
        "store":                 store_id,
        "product":               str(g("product_name", product_id)),
        "category":              category,
        "store_type":            str(g("store_type",   g("store_type_name", "?"))),
        "current_stock":         int(closing_stock),
        "incoming_stock":        int(incoming),
        "forecast_7d":           round(forecast_7d, 1),
        "forecast_p10":          round(p10, 1),
        "forecast_p90":          round(p90, 1),
        "stockout_probability":  round(stockout_prob, 4),
        "risk_level":            rl,
        "recommended_stock":     rq["recommended_stock"],
        "recommended_order":     rq["reorder_qty"],
        "safety_stock":          rq["safety_stock"],
        "adj_forecast":          rq["adj_forecast"],
        "expected_shortfall":    loss["expected_shortfall_units"],
        "expected_loss_inr":     loss["expected_loss_inr"],
        "margin_at_risk_inr":    loss["margin_at_risk_inr"],
        "top_reasons":           top_reasons_str,
        "manager_action":        action,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("src/recommend.py - self-test")

    # Test 1: reorder_qty is never negative
    for stock in [0, 50, 200, 500]:
        rq = reorder_qty(100, 20, 3.0, stock, 0, 30.0, 0.0)
        assert rq["reorder_qty"] >= 0, f"Negative reorder at stock={stock}"
    print("  PASS: reorder_qty never negative")

    # Test 2: HIGH / MEDIUM / LOW threshold boundaries
    assert risk_level(0.70) == "HIGH"
    assert risk_level(0.69) == "MEDIUM"
    assert risk_level(0.40) == "MEDIUM"
    assert risk_level(0.39) == "LOW"
    assert risk_level(0.00) == "LOW"
    print("  PASS: risk_level thresholds 0.70 / 0.40 correct")

    # Test 3: perishable cap reduces cover for milk (shelf_life=3)
    milk_cover = adjusted_forecast_cover(100.0, shelf_life_days=3.0)
    normal_cover = adjusted_forecast_cover(100.0, shelf_life_days=30.0)
    assert milk_cover < normal_cover, "Perishable cap should reduce cover"
    assert abs(milk_cover - 100.0 * (2.0 / 7.0)) < 0.01, f"Milk cover wrong: {milk_cover}"
    print(f"  PASS: perishable cap: milk cover {milk_cover:.1f} < normal {normal_cover:.1f}")

    # Test 4: reorder decreases when incoming stock increases
    rq_no_incoming  = reorder_qty(100, 20, 3.0, 10, 0,   30.0)
    rq_with_incoming = reorder_qty(100, 20, 3.0, 10, 50,  30.0)
    assert rq_no_incoming["reorder_qty"] >= rq_with_incoming["reorder_qty"], \
        "Reorder should decrease as incoming stock increases"
    print(f"  PASS: reorder with incoming=0: {rq_no_incoming['reorder_qty']} "
          f">= incoming=50: {rq_with_incoming['reorder_qty']}")

    # Test 5: overstock action
    action = manager_action("LOW", 0, days_of_cover=60.0, effective_lead=3.0)
    assert "Overstock" in action, f"Expected overstock action, got: {action}"
    print(f"  PASS: overstock action triggered: '{action}'")

    print("\nAll tests PASSED. src/recommend.py ready.")
