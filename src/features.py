"""
StockSense - Round 2 feature engineering (Person 2: ML Engineer).

Turns the Round 1 master table (one row per date x store x product) into a
modelling table with ONE ROW = ONE FORECAST ORIGIN: the end of day t for one
store x product. From that row both models predict the next 7 days (t+1..t+7):

    next_7_day_demand  - Model 1 target (sum of demand_est over t+1..t+7)
    stockout_next7     - Model 2 target (any stockout_flag over t+1..t+7)

Legality rule (the brief's leakage rule): every feature uses only what a store
knows at closing time on day t -
    * sales, stock and weather up to and including day t,
    * purchase orders placed by day t (their expected date, NOT when they really arrived),
    * calendars published in advance: planned promotions, holidays, festivals,
      local events and paydays for t+1..t+7.
`check_no_leakage()` proves it: it rebuilds the features from data truncated at
t and asserts they are identical.

Run from the repo root (after src/data_prep.py):
    python src/features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

COMBO = ["store_id", "product_id"]
HORIZON = 7
LAST_OBSERVED_DAY = pd.Timestamp("2026-08-31")
CALENDAR_END = pd.Timestamp("2026-09-07")
CAP_DAYS = 60  # "days since" features are capped: beyond two months the exact value carries no signal


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_inputs():
    m = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date", "launch_date"])
    promos = pd.read_csv(PROCESSED / "promotions_daily.csv", parse_dates=["date"])
    weather = pd.read_csv(PROCESSED / "external_factors_clean.csv", parse_dates=["date"])
    po = pd.read_csv(PROCESSED / "purchase_orders_clean.csv",
                     parse_dates=["order_date", "expected_date", "received_date"])
    return m, promos, weather, po


def roll(s, window, fn="mean", min_periods=1):
    return getattr(s.rolling(window, min_periods=min_periods), fn)()


def fwd(s, window=HORIZON, fn="sum"):
    """Aggregate over the NEXT `window` days (t+1..t+window); NaN if the window is incomplete."""
    r = getattr(s[::-1].rolling(window, min_periods=window), fn)()[::-1]
    return r.shift(-1)


def days_since(flag):
    """Days since `flag` was last 1, counting day t itself as 0; capped at CAP_DAYS."""
    idx = np.arange(len(flag))
    last = pd.Series(np.where(flag.values == 1, idx, np.nan), index=flag.index).ffill()
    return pd.Series(np.minimum(idx - last.values, CAP_DAYS), index=flag.index).fillna(CAP_DAYS)


# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------
def demand_features(m):
    """Level, weekly pattern, momentum and volatility of past demand (demand_est, not capped sales)."""
    g = m.groupby(COMBO, sort=False).demand_est
    f = pd.DataFrame(index=m.index)
    # lag_k = demand k days before the first forecast day t+1
    f["lag_1"] = m.demand_est
    f["lag_7"] = g.shift(6)
    f["lag_14"] = g.shift(13)
    for w in (7, 14, 28):
        f[f"rolling_mean_{w}"] = g.transform(roll, w)
    f["rolling_std_7"] = g.transform(roll, 7, "std", 2)
    f["rolling_std_28"] = g.transform(roll, 28, "std", 2)
    f["cv_28"] = f.rolling_std_28 / f.rolling_mean_28.replace(0, np.nan)
    f["prev_week_mean"] = g.shift(7).groupby([m.store_id, m.product_id], sort=False).transform(roll, 7)
    f["trend_7_28"] = f.rolling_mean_7 / f.rolling_mean_28.replace(0, np.nan)
    f["growth_wow"] = f.rolling_mean_7 / f.prev_week_mean.replace(0, np.nan)
    f["baseline_7d"] = 7 * f.rolling_mean_7   # the mandatory naive baseline, also a feature
    return f


def stock_features(m, po):
    """What is on the shelf, what is on the way, and how often this item ran out before."""
    g = m.groupby(COMBO, sort=False)
    f = pd.DataFrame(index=m.index)
    f["closing_stock"] = m.closing_stock
    f["reorder_level"] = m.reorder_level
    f["reorder_gap"] = m.closing_stock - m.reorder_level
    f["lead_time_days"] = m.lead_time_days

    f["incoming_next7"] = incoming_stock(m, po)
    rm7 = g.demand_est.transform(roll, 7)
    f["days_of_cover"] = ((m.closing_stock + f.incoming_next7) / rm7.clip(lower=0.5)).clip(upper=CAP_DAYS)
    f["stock_to_demand_7d"] = (m.closing_stock / (7 * rm7).clip(lower=1)).clip(upper=20)

    f["stockout_today"] = m.stockout_flag
    f["stockout_days_last14"] = g.stockout_flag.transform(roll, 14, "sum")
    f["stockout_rate_last28"] = g.stockout_flag.transform(roll, 28)
    f["days_since_stockout"] = g.stockout_flag.transform(days_since)
    f["lost_units_last7"] = g.lost_units_est.transform(roll, 7, "sum")
    return f


def incoming_stock(m, po):
    """
    Units on open purchase orders that are due by t+7, as the store sees them at close of day t.
    A PO counts from its order date until the day before it actually arrives; the store knows
    the EXPECTED date, so overdue POs still count as incoming (the store is still waiting for them).
    """
    p = po[po.qty_ordered > 0].copy()
    start = np.maximum(p.order_date, p.expected_date - pd.Timedelta(days=HORIZON))
    end = (p.received_date - pd.Timedelta(days=1)).fillna(LAST_OBSERVED_DAY)
    p = p.assign(start=start, end=end)
    p = p[p.end >= p.start]
    p["date"] = [pd.date_range(s, e) for s, e in zip(p.start, p.end)]
    p = p.explode("date")
    inc = p.groupby(COMBO + ["date"]).qty_ordered.sum()
    key = pd.MultiIndex.from_frame(m[COMBO + ["date"]])
    return pd.Series(inc.reindex(key).fillna(0).values, index=m.index)


def supplier_features(m, po):
    """Supplier reliability as known on day t: expanding late rate / delay over POs already due by t."""
    rec = po[po.status == "RECEIVED"].copy()
    days = pd.date_range(m.date.min(), LAST_OBSERVED_DAY)
    rows = []
    for sup, d in rec.groupby("supplier_id"):
        # a PO's lateness is known once it has arrived (or once it is overdue - approximated by arrival)
        known = d.groupby("received_date").agg(n=("po_id", "size"), late=("is_late", "sum"),
                                              delay=("days_late", "sum"), fill=("fill_rate", "sum"))
        known = known.reindex(days, fill_value=0).cumsum()
        rows.append(pd.DataFrame({
            "supplier_id": sup, "date": days,
            "supplier_late_rate": (known.late / known.n.replace(0, np.nan)).values,
            "supplier_avg_delay": (known.delay / known.n.replace(0, np.nan)).values,
            "supplier_fill_rate": (known.fill / known.n.replace(0, np.nan)).values,
        }))
    s = pd.concat(rows)
    out = m[["supplier_id", "date"]].merge(s, on=["supplier_id", "date"], how="left")
    out.index = m.index
    out["effective_lead_time"] = m.lead_time_days + out.supplier_avg_delay.fillna(0)
    return out.drop(columns=["supplier_id", "date"])


def promo_features(m, promos):
    """Planned promotions (from the calendar) in the forecast window, plus recent promotion history."""
    combos = m.groupby(COMBO).launch_date.min().reset_index()
    grid = pd.concat([pd.DataFrame({"store_id": r.store_id, "product_id": r.product_id,
                                    "date": pd.date_range(r.launch_date, CALENDAR_END)})
                      for r in combos.itertuples()], ignore_index=True)
    grid = grid.merge(promos[COMBO + ["date", "discount_pct"]], on=COMBO + ["date"], how="left")
    grid["disc"] = grid.discount_pct.fillna(0) / 100
    grid["on"] = (grid.disc > 0).astype(int)
    g = grid.groupby(COMBO, sort=False)
    grid["promo_days_next7"] = g.on.transform(fwd)
    grid["max_discount_next7"] = g.disc.transform(fwd, fn="max")
    grid["avg_discount_next7"] = g.disc.transform(fwd, fn="mean")
    grid["promo_today"] = grid.on
    grid["promo_days_last7"] = g.on.transform(roll, 7, "sum")
    grid["days_since_promo"] = g.on.transform(days_since)
    grid["post_promo_window"] = ((grid.promo_days_last7 > 0) & (grid.promo_days_next7 == 0)).astype(int)
    cols = ["promo_days_next7", "max_discount_next7", "avg_discount_next7", "promo_today",
            "promo_days_last7", "days_since_promo", "post_promo_window"]
    out = m[COMBO + ["date"]].merge(grid[COMBO + ["date"] + cols], on=COMBO + ["date"], how="left")
    out.index = m.index
    return out[cols]


def calendar_features(m, weather):
    """Holidays, festivals, local events and paydays in t+1..t+7 (published in advance) + recent weather."""
    w = weather.sort_values(["city", "date"]).copy()
    w["payday"] = (w.date.dt.day <= 5).astype(int)
    g = w.groupby("city", sort=False)
    w["holiday_next7"] = g.holiday.transform(fwd)
    w["festival_next7"] = g.festival.transform(fwd, fn="max")
    w["local_event_next7"] = g.local_event.transform(fwd)
    w["payday_days_next7"] = g.payday.transform(fwd)
    # Past weather only. Future weather rows exist but would stand in for a weather forecast.
    w["temp_last3"] = g.temp_c.transform(roll, 3)
    w["rain_last3"] = g.rain_mm.transform(roll, 3, "sum")
    w["temp_anomaly"] = w.temp_last3 - g.temp_c.transform(roll, 28)

    fest_dates = np.sort(w.loc[w.festival == 1, "date"].unique())
    nxt = np.searchsorted(fest_dates, w.date.values, side="right")
    w["days_to_festival"] = np.where(nxt < len(fest_dates),
                                     (fest_dates[np.minimum(nxt, len(fest_dates) - 1)] - w.date.values)
                                     / np.timedelta64(1, "D"), 30).clip(max=30)
    cols = ["holiday_next7", "festival_next7", "local_event_next7", "payday_days_next7", "days_to_festival",
            "temp_last3", "rain_last3", "temp_anomaly"]
    out = m[["city", "date"]].merge(w[["city", "date"] + cols], on=["city", "date"], how="left")
    out.index = m.index
    out["origin_dow"] = m.date.dt.dayofweek
    out["month"] = m.date.dt.month
    return out[cols + ["origin_dow", "month"]]


def price_features(m):
    """Regular (non-promotional) shelf price and how it moved - several MRP hikes happen mid-period."""
    reg = m.unit_price.where(m.discount_pct == 0)
    reg = reg.groupby([m.store_id, m.product_id], sort=False).ffill().fillna(m.mrp)
    g = reg.groupby([m.store_id, m.product_id], sort=False)
    f = pd.DataFrame(index=m.index)
    f["regular_price"] = reg
    f["price_ratio_28"] = reg / g.transform(roll, 28, "median")
    f["price_change_30d"] = reg / g.shift(30).fillna(g.transform("first")) - 1
    f["discount_today"] = m.discount_pct / 100
    return f


def product_festival_profile(m, weather):
    """
    How much each product's demand rises in the 3 days before a festival, as known on day t:
    an expanding estimate over festivals that have already happened, pooled over stores and
    shrunk towards 1 (no effect) while evidence is thin. Returns one value per master row.
    """
    fest = np.sort(weather.loc[weather.festival == 1, "date"].unique())   # published calendar
    to_fest = np.full(len(m), 99.0)
    for d in fest:
        gap = (d - m.date.values) / np.timedelta64(1, "D")
        to_fest = np.where((gap >= 0) & (gap < to_fest), gap, to_fest)
    near = to_fest <= 3
    # each day relative to the item's own recent level, so big stores do not dominate
    level = m.groupby(COMBO, sort=False).demand_est.transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    rel = (m.demand_est / level.replace(0, np.nan)).clip(upper=10)
    d = pd.DataFrame({"product_id": m.product_id, "date": m.date, "rel": rel, "near": near}).dropna(subset=["rel"])
    agg = (d.assign(near_sum=d.rel.where(d.near, 0), near_n=d.near.astype(int),
                    far_sum=d.rel.where(~d.near, 0), far_n=(~d.near).astype(int))
           .groupby(["product_id", "date"])[["near_sum", "near_n", "far_sum", "far_n"]].sum()
           .groupby(level="product_id").cumsum())
    k = 20  # prior strength in "festival-window observations"
    raw = (agg.near_sum / agg.near_n.replace(0, np.nan)) / (agg.far_sum / agg.far_n.replace(0, np.nan))
    lift = ((raw.fillna(1.0) * agg.near_n + k) / (agg.near_n + k)).rename("product_festival_lift")
    key = pd.MultiIndex.from_frame(m[["product_id", "date"]])
    return pd.Series(lift.reindex(key).values, index=m.index).fillna(1.0)


def static_features(m, fest_lift):
    f = pd.DataFrame(index=m.index)
    f["store_code"] = m.store_id.str[1:].astype(int)
    f["store_type"] = m.store_type
    f["category"] = m.category
    f["sub_category"] = m.sub_category
    f["shelf_life_days"] = m.shelf_life_days
    f["mrp"] = m.mrp
    f["avg_daily_customers"] = m.avg_daily_customers
    f["floor_area_sqft"] = m.floor_area_sqft
    f["history_days"] = m.history_days
    f["product_festival_lift"] = fest_lift
    f["festival_x_lift"] = 0.0  # filled after calendar features are joined
    return f


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def targets(m):
    g = m.groupby(COMBO, sort=False)
    t = pd.DataFrame(index=m.index)
    t["next_7_day_demand"] = g.demand_est.transform(fwd)
    t["next_7_day_units_sold"] = g.units_sold.transform(fwd)          # censored version, for comparison
    t["stockout_next7"] = g.stockout_flag.transform(fwd, fn="max")    # Model 2 target (Student 3)
    t["stockout_days_next7"] = g.stockout_flag.transform(fwd)
    t["lost_units_next7"] = g.lost_units_est.transform(fwd)
    return t


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
ID_COLS = ["date", "store_id", "product_id", "product_name", "category", "store_type", "city", "supplier_id",
           "sparse_history_flag", "unit_price", "cost_price"]
CATEGORICAL = ["store_type", "category", "sub_category"]
TARGETS = ["next_7_day_demand", "next_7_day_units_sold", "stockout_next7", "stockout_days_next7", "lost_units_next7"]


def build_features(m, promos, weather, po):
    m = m.sort_values(COMBO + ["date"]).reset_index(drop=True)
    parts = [
        demand_features(m),
        stock_features(m, po),
        supplier_features(m, po),
        promo_features(m, promos),
        calendar_features(m, weather),
        price_features(m),
        static_features(m, product_festival_profile(m, weather)),
    ]
    X = pd.concat(parts, axis=1)
    X["festival_x_lift"] = X.festival_next7 * (X.product_festival_lift - 1)
    ids = m[ID_COLS].rename(columns={"category": "category_name", "store_type": "store_type_name"})
    out = pd.concat([ids, X, targets(m)], axis=1)
    assert len(out) == len(m) and not out.duplicated(["date"] + COMBO).any()
    return out


FEATURE_GROUPS = {
    "Recent demand": ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
                      "rolling_std_7", "rolling_std_28", "cv_28", "prev_week_mean", "trend_7_28",
                      "growth_wow", "baseline_7d"],
    "Promotion plan": ["promo_days_next7", "max_discount_next7", "avg_discount_next7", "promo_today",
                       "promo_days_last7", "days_since_promo", "post_promo_window", "discount_today"],
    "Festival & holiday calendar": ["holiday_next7", "festival_next7", "days_to_festival", "product_festival_lift",
                                    "festival_x_lift", "local_event_next7", "payday_days_next7"],
    "Weather": ["temp_last3", "rain_last3", "temp_anomaly"],
    "Price": ["regular_price", "price_ratio_28", "price_change_30d", "mrp"],
    "Stock & supply": ["closing_stock", "reorder_level", "reorder_gap", "incoming_next7", "days_of_cover",
                       "stock_to_demand_7d", "stockout_today", "stockout_days_last14", "stockout_rate_last28",
                       "days_since_stockout", "lost_units_last7", "lead_time_days", "supplier_late_rate",
                       "supplier_avg_delay", "supplier_fill_rate", "effective_lead_time"],
    "Store & product profile": ["store_code", "store_type", "category", "sub_category", "shelf_life_days",
                                "avg_daily_customers", "floor_area_sqft", "history_days"],
    "Calendar position": ["origin_dow", "month"],
}
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]


# --------------------------------------------------------------------------
# Leakage proof
# --------------------------------------------------------------------------
def check_no_leakage(m, promos, weather, po, full, origins=("2026-04-20", "2026-06-10", "2026-07-25", "2026-08-24")):
    """
    For several origin dates t: throw away every observation after t (sales, stock, weather,
    PO arrivals), rebuild the features, and assert they equal the full-data features at t.
    Calendars of planned promotions / holidays stay, because they are known in advance.
    """
    checked = 0
    for o in map(pd.Timestamp, origins):
        mt = m[m.date <= o]
        pt = po[po.order_date <= o].copy()
        not_yet = pt.received_date > o
        pt.loc[not_yet, ["received_date", "qty_received", "days_late", "is_late", "fill_rate"]] = np.nan
        pt.loc[not_yet, "status"] = "OPEN"
        wt = weather.copy()
        wt.loc[wt.date > o, ["temp_c", "rain_mm"]] = np.nan   # future weather unknown
        ft = build_features(mt, promos, wt, pt)
        a = ft[ft.date == o].set_index(COMBO)[FEATURES]
        b = full[full.date == o].set_index(COMBO).loc[a.index, FEATURES]
        for col in FEATURES:
            x, y = a[col], b[col]
            if not pd.api.types.is_numeric_dtype(x):
                assert (x == y).all(), f"leak in {col} at {o.date()}"
            else:
                assert np.allclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9, atol=1e-9), \
                    f"leak in {col} at {o.date()}"
        checked += len(a)
    return checked


def write_feature_dictionary(full):
    why = {
        "Recent demand": "Level, weekly pattern, momentum and volatility. Built on demand_est (sales corrected for empty shelves).",
        "Promotion plan": "From the promotions calendar (planned in advance), never from future transactions. Captures lift and the post-promotion dip.",
        "Festival & holiday calendar": "Festivals drive pre-festival buying (pooja items, ghee, sweets). Calendars are published in advance.",
        "Weather": "Heat drives beverages and ice cream. Past 3 days only; future weather would be a forecast.",
        "Price": "Several MRP hikes happen mid-period; demand falls after a hike.",
        "Stock & supply": "What is on the shelf and on the way (open POs by expected date). Mainly for Model 2; supplier reliability is an expanding rate over POs already received.",
        "Store & product profile": "Lets one global model share learning across 387 combinations and handle new products.",
        "Calendar position": "Where the forecast window sits in the week / season.",
    }
    legal = {
        "Recent demand": "Days <= t", "Promotion plan": "Planned calendar for t+1..t+7; history <= t",
        "Festival & holiday calendar": "Published calendar; festival lift from festivals already past at t",
        "Weather": "Days <= t", "Price": "Days <= t", "Stock & supply": "Closing stock of t; POs ordered <= t (expected date)",
        "Store & product profile": "Static", "Calendar position": "Known",
    }
    rows = []
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            s = full[c]
            miss = s.isna().mean()
            rows.append((grp, c, legal[grp], f"{miss:.1%}"))
    d = pd.DataFrame(rows, columns=["group", "feature", "information used", "missing"])
    d.to_csv(REPORTS / "feature_dictionary.csv", index=False)
    lines = ["# Feature dictionary - Model 1 (demand) and Model 2 (stock-out)", "",
             "One row = one forecast origin: close of day *t* for one store x product. Both models predict *t+1..t+7*.",
             "",
             "**Leakage rule.** Every feature uses only information available at close of day *t*. "
             "`check_no_leakage()` in `src/features.py` rebuilds the features from data cut off at *t* "
             "(sales, stock, weather and PO arrivals after *t* removed) and asserts they are identical.",
             "",
             "Missing values are left as NaN on purpose: they occur only where there is no history yet "
             "(new products, the first days of the period). XGBoost handles NaN natively; the other models impute.",
             "",
             "| Group | Why it matters | Information used |", "|---|---|---|"]
    lines += [f"| {g} | {why[g]} | {legal[g]} |" for g in FEATURE_GROUPS]
    lines += ["", "## Every feature", "", "| Group | Feature | Missing |", "|---|---|---|"]
    lines += [f"| {r.group} | `{r.feature}` | {r.missing} |" for r in d.itertuples()]
    lines += ["", "## Targets", "",
              "| Target | Definition | Used by |", "|---|---|---|",
              "| `next_7_day_demand` | Sum of `demand_est` over t+1..t+7 | Model 1 (Student 2) |",
              "| `next_7_day_units_sold` | Sum of `units_sold` over t+1..t+7 (censored by stock-outs) | Comparison only |",
              "| `stockout_next7` | 1 if closing stock hits 0 on any day in t+1..t+7 | Model 2 (Student 3) |",
              "| `stockout_days_next7`, `lost_units_next7` | Count / lost units in the window | Model 2 analysis, prioritisation |",
              "",
              "Dropped on purpose: `weekend_days_next7` - every 7-day window contains exactly 2 weekend days, so it is constant."]
    (REPORTS / "feature_dictionary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading processed data ...")
    m, promos, weather, po = load_inputs()
    print("Building features ...")
    full = build_features(m, promos, weather, po)
    print("Checking for leakage ...")
    n = check_no_leakage(m, promos, weather, po, full)
    print(f"  {n} origin rows rebuilt from truncated data: all {len(FEATURES)} features identical")
    full.to_csv(PROCESSED / "features.csv", index=False)
    write_feature_dictionary(full)
    has_target = full.next_7_day_demand.notna()
    print(f"features.csv: {len(full):,} origin rows x {len(FEATURES)} features "
          f"({has_target.sum():,} with a complete 7-day target; origins {full.date.min().date()} .. {full.date.max().date()})")
    print(f"stockout_next7 positive rate: {full.stockout_next7.mean():.1%}")
    return full


if __name__ == "__main__":
    main()
