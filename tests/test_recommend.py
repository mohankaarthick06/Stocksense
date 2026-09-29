"""
StockSense - Round 3 (Student 3: Decision Intelligence)
tests/test_recommend.py - Unit tests for the business action layer

Run: python tests/test_recommend.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommend import (
    adjusted_forecast_cover,
    expected_loss_inr,
    manager_action,
    reorder_qty,
    risk_level,
    safety_stock,
)


def test_reorder_never_negative():
    """Reorder qty must be >= 0 regardless of stock level."""
    for stock in [0, 10, 50, 100, 200, 500, 1000]:
        rq = reorder_qty(100, 20, 3.0, stock, 0, 30.0)
        assert rq["reorder_qty"] >= 0, f"Negative reorder at stock={stock}"
    # Excess incoming also must not produce negative reorder
    rq2 = reorder_qty(100, 20, 3.0, 0, 9999, 30.0)
    assert rq2["reorder_qty"] == 0, f"Reorder should be 0 with huge incoming, got {rq2['reorder_qty']}"
    print("  PASS: reorder_qty never negative")


def test_risk_level_thresholds():
    """HIGH >= 0.70, MEDIUM 0.40-0.70, LOW < 0.40."""
    assert risk_level(1.00) == "HIGH"
    assert risk_level(0.70) == "HIGH"
    assert risk_level(0.699) == "MEDIUM"
    assert risk_level(0.50) == "MEDIUM"
    assert risk_level(0.40) == "MEDIUM"
    assert risk_level(0.399) == "LOW"
    assert risk_level(0.00) == "LOW"
    print("  PASS: risk_level thresholds 0.70 / 0.40 correct")


def test_perishable_cap_milk():
    """For shelf_life=3, cover = (3-1)/7 * forecast = 2/7 * 100 ~= 28.57."""
    cover = adjusted_forecast_cover(100.0, shelf_life_days=3.0)
    expected = 100.0 * (2.0 / 7.0)
    assert abs(cover - expected) < 0.01, f"Milk cover {cover:.2f} != expected {expected:.2f}"
    # Non-perishable should be unchanged
    normal = adjusted_forecast_cover(100.0, shelf_life_days=30.0)
    assert abs(normal - 100.0) < 0.01, f"Non-perishable cover {normal:.2f} != 100.0"
    print(f"  PASS: perishable cap: milk {cover:.1f} < normal {normal:.1f}")


def test_reorder_decreases_with_incoming():
    """Adding more incoming stock should reduce (or keep equal) the reorder quantity."""
    rq_no  = reorder_qty(100, 20, 3.0, 10, 0, 30.0)["reorder_qty"]
    rq_med = reorder_qty(100, 20, 3.0, 10, 30, 30.0)["reorder_qty"]
    rq_big = reorder_qty(100, 20, 3.0, 10, 100, 30.0)["reorder_qty"]
    assert rq_no >= rq_med >= rq_big, (
        f"Reorder should decrease with incoming: {rq_no} >= {rq_med} >= {rq_big}"
    )
    print(f"  PASS: reorder decreases with incoming: {rq_no} -> {rq_med} -> {rq_big}")


def test_overstock_action():
    """days_of_cover > 2*(lead+7) should trigger Overstock action."""
    action = manager_action("LOW", 0, days_of_cover=60.0, effective_lead=3.0)
    assert "Overstock" in action, f"Expected Overstock action, got: {action}"
    # Not overstock at normal cover
    action2 = manager_action("LOW", 0, days_of_cover=10.0, effective_lead=3.0)
    assert "Overstock" not in action2, f"Should not be overstock: {action2}"
    print(f"  PASS: overstock action triggered correctly")


def test_decision_matrix():
    """Check all 6 cells of the risk x reorder decision matrix."""
    assert manager_action("HIGH",   10, 5.0, 3.0) == "Raise replenishment order today"
    assert manager_action("HIGH",    0, 5.0, 3.0) == "Expedite / confirm open PO"
    assert manager_action("MEDIUM", 10, 5.0, 3.0) == "Add to next order cycle"
    assert manager_action("MEDIUM",  0, 5.0, 3.0) == "Monitor daily"
    assert manager_action("LOW",    10, 5.0, 3.0) == "Order at normal cycle"
    assert manager_action("LOW",     0, 5.0, 3.0) == "No action"
    print("  PASS: all 6 decision matrix cells correct")


def test_expected_loss():
    """Loss = prob * max(0, forecast - stock - incoming) * price."""
    loss = expected_loss_inr(0.8, 100.0, 20.0, 10.0, 50.0)
    shortfall = max(0, 100 - 20 - 10)   # = 70
    expected  = 0.8 * 70 * 50.0          # = 2800
    assert abs(loss["expected_loss_inr"] - expected) < 0.01, \
        f"Loss {loss['expected_loss_inr']} != {expected}"
    # No loss when stock covers demand
    loss2 = expected_loss_inr(0.9, 50.0, 60.0, 20.0, 50.0)
    assert loss2["expected_loss_inr"] == 0.0, "No loss when well-stocked"
    print(f"  PASS: expected_loss_inr correct: {loss['expected_loss_inr']:.2f}")


def test_safety_stock_z():
    """Perishables use z=1.28; non-perishables use z=1.65."""
    ss_peri   = safety_stock(20.0, 3.0, shelf_life_days=3.0)
    ss_normal = safety_stock(20.0, 3.0, shelf_life_days=30.0)
    assert ss_peri < ss_normal, "Perishable safety stock should be lower"
    print(f"  PASS: safety stock: perishable {ss_peri:.1f} < normal {ss_normal:.1f}")


def test_bias_adjustment():
    """Positive bias_units should reduce adj_forecast (model over-forecasted)."""
    rq_no_bias = reorder_qty(100, 20, 3.0, 10, 0, 30.0, bias_units=0.0)
    rq_pos_bias = reorder_qty(100, 20, 3.0, 10, 0, 30.0, bias_units=10.0)
    assert rq_pos_bias["adj_forecast"] < rq_no_bias["adj_forecast"], \
        "Positive bias should reduce adjusted forecast"
    print(f"  PASS: bias adjustment: {rq_no_bias['adj_forecast']} -> {rq_pos_bias['adj_forecast']}")


if __name__ == "__main__":
    print("Running unit tests for src/recommend.py\n")
    tests = [
        test_reorder_never_negative,
        test_risk_level_thresholds,
        test_perishable_cap_milk,
        test_reorder_decreases_with_incoming,
        test_overstock_action,
        test_decision_matrix,
        test_expected_loss,
        test_safety_stock_z,
        test_bias_adjustment,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if failed == 0:
        print(f"All {len(tests)} tests PASSED")
    else:
        print(f"{failed}/{len(tests)} tests FAILED")
