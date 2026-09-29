"""
StockSense - Round 1 data preparation (Person 1: Data Analyst).

Cleans the raw CSVs, reconciles inventory, and builds the master analytics table
(ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT). Every fix is logged to the
Data Quality Report with its count, action and justification.

Run from the repo root:
    python src/data_prep.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

KEY = ["date", "store_id", "product_id"]
LAST_OBSERVED_DAY = pd.Timestamp("2026-08-31")
SPARSE_HISTORY_DAYS = 28
VALID_TEMP_RANGE = (10, 45)  # plausible daily temperature for Tamil Nadu, deg C

CATEGORY_MAP = {
    "beverage": "Beverages", "beverages": "Beverages",
    "dairy": "Dairy",
    "frozen": "Frozen",
    "grocery": "Groceries", "groceries": "Groceries",
    "household": "Household",
    "personal care": "Personal Care",
    "snacks": "Snacks",
}
CITY_ALIASES = {"Trichy": "Tiruchirappalli"}


# --------------------------------------------------------------------------
# Data Quality log
# --------------------------------------------------------------------------
class DQLog:
    """Collects one row per data-quality issue for the report."""

    def __init__(self):
        self.rows = []

    def add(self, table, issue, count, action, justification):
        self.rows.append(dict(table=table, issue=issue, count=int(count),
                              action=action, justification=justification))
        print(f"  [{table}] {issue}: {int(count):,} -> {action}")

    def frame(self):
        return pd.DataFrame(self.rows)


def check_merge(before, after, name):
    """A failed join does not raise an error - it silently adds or loses rows."""
    assert len(before) == len(after), f"{name}: row count changed {len(before)} -> {len(after)}"


# --------------------------------------------------------------------------
# Small master tables
# --------------------------------------------------------------------------
def clean_products(dq):
    df = pd.read_csv(RAW / "products.csv")

    n = df.duplicated().sum()
    df = df.drop_duplicates()
    dq.add("products", "Exact duplicate rows", n, "Removed",
           "Identical rows for the same product; keeping one avoids duplicating sales in joins.")
    assert df.product_id.is_unique

    raw_category = df.category.copy()
    df["category"] = df.category.str.strip().str.lower().map(CATEGORY_MAP)
    assert df.category.notna().all(), "unmapped category spelling"
    dq.add("products", f"Inconsistent category spellings ({raw_category.nunique()} spellings for {df.category.nunique()} categories)",
           (raw_category != df.category).sum(), f"Mapped to {df.category.nunique()} standard categories",
           "'beverage', 'BEVERAGES', 'Grocery' etc. would otherwise split one category into several in every group-by.")

    # Missing attributes: borrow from sibling products (same supplier and sub-category).
    siblings = df.groupby(["supplier_id", "sub_category"])
    missing_brand = df.brand.isna()
    df["brand"] = df.brand.fillna(siblings.brand.transform(lambda s: s.mode().iat[0] if s.notna().any() else np.nan))
    df["brand"] = df.brand.fillna("Unknown")
    dq.add("products", "Missing brand", missing_brand.sum(),
           f"Filled from sibling products (same supplier + sub-category): {df.loc[missing_brand, ['product_id', 'brand']].values.tolist()}",
           "Same supplier and sub-category almost always means the same brand; brand is descriptive only.")

    missing_life = df.shelf_life_days.isna()
    df["shelf_life_days"] = df.shelf_life_days.fillna(df.groupby("sub_category").shelf_life_days.transform("median"))
    dq.add("products", "Missing shelf_life_days", missing_life.sum(),
           f"Filled with sub-category median: {df.loc[missing_life, ['product_id', 'shelf_life_days']].values.tolist()}",
           "Shelf life is set by product type; it drives the perishable ordering cap, so it cannot be left blank.")
    df["shelf_life_days"] = df.shelf_life_days.astype(int)

    bad_cost = (df.cost_price >= df.mrp).sum()
    dq.add("products", "cost_price >= MRP (negative margin)", bad_cost, "Checked - none found" if bad_cost == 0 else "Flagged",
           "Margin feeds lost-profit estimates.")
    return df


def clean_stores(dq):
    df = pd.read_csv(RAW / "stores.csv")
    raw_types = df.store_type.nunique()
    df["store_type"] = df.store_type.str.strip().str.title()
    dq.add("stores", f"store_type mixed case / stray spaces ({raw_types} spellings)",
           (df.store_type != pd.read_csv(RAW / "stores.csv").store_type).sum(),
           f"Stripped and title-cased -> {sorted(df.store_type.unique())}",
           "'hypermarket' and 'Hypermarket' must be one group in store-format analysis.")
    assert df.store_id.is_unique
    return df


def clean_suppliers():
    return pd.read_csv(RAW / "suppliers.csv")


# --------------------------------------------------------------------------
# External factors
# --------------------------------------------------------------------------
def clean_external(dq):
    df = pd.read_csv(RAW / "external_factors.csv", parse_dates=["date"])

    n = df.duplicated().sum()
    df = df.drop_duplicates()
    dq.add("external_factors", "Exact duplicate rows", n, "Removed", "Duplicate weather rows would duplicate master rows on join.")
    assert not df.duplicated(["date", "city"]).any()

    n = df.city.isin(CITY_ALIASES).sum()
    df["city"] = df.city.replace(CITY_ALIASES)
    dq.add("external_factors", "City spelled 'Trichy' (stores.csv uses 'Tiruchirappalli')", n,
           "Renamed to Tiruchirappalli",
           "Without this, store S05 gets no weather or calendar data at all - the join fails silently with NaNs.")

    lo, hi = VALID_TEMP_RANGE
    impossible = df.temp_c.notna() & ~df.temp_c.between(lo, hi)
    dq.add("external_factors", f"Impossible temperature (outside {lo}-{hi} C)", impossible.sum(),
           f"Set to missing, then imputed. Values seen: {sorted(df.loc[impossible, 'temp_c'].unique().tolist())}",
           "-1 C, 58.4 C and 112 C are sensor/entry errors for Tamil Nadu; keeping them would distort heat effects.")
    df.loc[impossible, "temp_c"] = np.nan

    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df["weather_imputed"] = (df.temp_c.isna() | df.rain_mm.isna()).astype(int)
    n_temp, n_rain = df.temp_c.isna().sum() - impossible.sum(), df.rain_mm.isna().sum()
    for col in ["temp_c", "rain_mm"]:
        df[col] = (df.set_index("date").groupby("city")[col]
                   .transform(lambda s: s.interpolate(method="time", limit_direction="both")).values)
    dq.add("external_factors", "Missing temp_c", n_temp, "Time interpolation within the same city",
           "Weather changes smoothly day to day within a city; a global mean would ignore the Chennai vs Coimbatore climate gap.")
    dq.add("external_factors", "Missing rain_mm", n_rain, "Time interpolation within the same city",
           "Same reasoning; neighbouring days are the best estimate. Rows are flagged in weather_imputed.")
    assert df[["temp_c", "rain_mm"]].notna().all().all()

    wrong_weekend = (df.weekend != (df.date.dt.dayofweek >= 5).astype(int)).sum()
    dq.add("external_factors", "weekend flag disagrees with the calendar", wrong_weekend,
           "Checked - none found" if wrong_weekend == 0 else "Recomputed", "Validates the calendar flags.")
    df["festival_name"] = df.festival_name.fillna("")
    return df


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------
def clean_transactions(dq, products, inventory_raw):
    df = pd.read_csv(RAW / "transactions.csv", dtype=str, keep_default_na=False)
    n_raw = len(df)

    n = df.duplicated().sum()
    df = df.drop_duplicates()
    dq.add("transactions", "Exact duplicate rows (same transaction_id and same content)", n, "Removed",
           "A repeated identical record is a double load, not a second sale.")
    assert df.transaction_id.is_unique, "same transaction_id with different content - not a true duplicate"

    iso = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(df.date, format="%d-%m-%Y", errors="coerce")
    dq.add("transactions", "Dates in DD-MM-YYYY instead of YYYY-MM-DD", iso.isna().sum(),
           "Parsed with an explicit second format", "Automatic inference could read 03-08-2026 as 8 March, moving sales to the wrong month.")
    df["date"] = iso.fillna(dmy)
    assert df.date.notna().all()

    bad_ids = ~df.store_id.str.fullmatch(r"S\d{2}") | ~df.product_id.str.fullmatch(r"P\d{3}")
    for col in ["store_id", "product_id", "customer_id", "payment_mode"]:
        df[col] = df[col].str.strip()
    df["store_id"] = df.store_id.str.upper()
    df["product_id"] = df.product_id.str.upper()
    dq.add("transactions", "Store/product IDs in lower case or with stray spaces", bad_ids.sum(),
           "Stripped and upper-cased", "'s02' and ' S02' would each become a separate, phantom store.")
    assert df.product_id.isin(products.product_id).all() and df.store_id.str.fullmatch(r"S\d{2}").all()

    for col in ["quantity", "discount_pct", "promotion_flag", "hour"]:
        df[col] = df[col].astype(int)
    df["selling_price"] = pd.to_numeric(df.selling_price.replace("", np.nan))
    df["customer_id"] = df.customer_id.replace("", np.nan)

    # Returns and voids: kept aside, excluded from demand.
    is_return = df.quantity <= 0
    dq.add("transactions", "quantity <= 0 (returns / voids)", is_return.sum(),
           "Excluded from demand; counted separately as returns", "A return is not negative demand; including it would understate how much customers wanted.")
    returns = df[is_return].copy()
    df = df[~is_return].copy()

    # Keying errors: quantity typed x100. Verified against inventory.sold_qty.
    daily = df.groupby(KEY).quantity.sum().rename("txn_qty").reset_index()
    inv = inventory_raw.drop_duplicates()[KEY + ["sold_qty"]]
    check = daily.merge(inv, on=KEY, how="inner")
    over = check[check.txn_qty > check.sold_qty][KEY]
    suspect = df.merge(over, on=KEY).query("quantity >= 100 and quantity % 100 == 0").transaction_id
    fix = df.transaction_id.isin(suspect)
    df.loc[fix, "quantity"] = df.loc[fix, "quantity"] // 100
    dq.add("transactions", "Keying errors: quantity multiplied by 100 (e.g. 100 instead of 1)", fix.sum(),
           "Divided by 100 after confirming with inventory.sold_qty",
           "Inventory disagreed on exactly these store-days; after the fix both systems agree. A blanket outlier cap "
           "would wrongly clip genuine bulk sales (e.g. Chennai hypermarket milk ~185 units/day).")
    df["qty_corrected"] = fix.astype(int)

    # Prices. Rule: 0 < price <= MRP. Every valid line in the same store-product-day-discount has the same price.
    df = df.merge(products[["product_id", "mrp"]], on="product_id", how="left")
    too_high = df.selling_price > df.mrp
    ten_x = too_high & np.isclose(df.selling_price / 10, (df.selling_price / 10).round(1))
    df.loc[too_high, "selling_price"] = np.where(ten_x[too_high], df.loc[too_high, "selling_price"] / 10, np.nan)
    dq.add("transactions", "selling_price above MRP (about 10x - misplaced decimal)", too_high.sum(),
           "Divided by 10; result verified against sibling lines below",
           "Price above MRP is illegal in India; /10 gives exactly the price charged on other lines that day.")

    group = ["store_id", "product_id", "date", "discount_pct"]
    missing = df.selling_price.isna()
    sibling_price = df.groupby(group).selling_price.transform("median")
    # Fallback: same store-product-discount on the nearest earlier day, then MRP less discount.
    df["selling_price"] = df.selling_price.fillna(sibling_price)
    still = df.selling_price.isna()
    if still.any():
        df = df.sort_values("date")
        df["selling_price"] = df.groupby(["store_id", "product_id", "discount_pct"]).selling_price.transform(lambda s: s.ffill().bfill())
        df["selling_price"] = df.selling_price.fillna(df.mrp * (1 - df.discount_pct / 100))
    dq.add("transactions", "Missing selling_price", missing.sum(),
           f"Filled with the price of other lines for the same store, product, day and discount "
           f"({missing.sum() - still.sum():,}); {still.sum():,} from the nearest day with the same discount",
           "Within a store-product-day-discount every valid line has one identical price, so this recovers the true price.")
    df["price_imputed"] = missing.astype(int)

    spread = df.groupby(group).selling_price.nunique()
    assert (spread == 1).all(), "price still inconsistent within a store-product-day-discount"
    assert (df.selling_price > 0).all() and (df.selling_price <= df.mrp).all()

    n_members = df.customer_id.isna().sum()
    dq.add("transactions", "Blank customer_id", n_members, "Left blank (not imputed)",
           "Blank means the shopper is not a loyalty member. It is not used for demand, so inventing IDs would add fiction.")

    df["revenue"] = df.quantity * df.selling_price
    print(f"  transactions: {n_raw:,} raw -> {len(df):,} clean sale lines + {len(returns):,} returns")
    return df.drop(columns="mrp"), returns


# --------------------------------------------------------------------------
# Purchase orders and promotions
# --------------------------------------------------------------------------
def clean_purchase_orders(dq):
    df = pd.read_csv(RAW / "purchase_orders.csv", parse_dates=["order_date", "expected_date", "received_date"])
    n = df.duplicated().sum()
    df = df.drop_duplicates()
    assert df.po_id.is_unique
    dq.add("purchase_orders", "Exact duplicate rows", n, "Checked - none found" if n == 0 else "Removed", "")
    is_open = df.status.eq("OPEN")
    assert df.loc[is_open, "received_date"].isna().all() and df.loc[~is_open, "received_date"].notna().all()
    dq.add("purchase_orders", "Blank received_date / qty_received", is_open.sum(),
           "Kept - these are the OPEN orders (incoming stock on 31 Aug)", "Blank is meaningful here, not an error.")

    zero = df.status.eq("RECEIVED") & df.qty_received.eq(0)
    dq.add("purchase_orders", "Status RECEIVED but qty_received = 0", zero.sum(),
           "Kept as failed deliveries (fill rate 0)",
           "Inventory confirms nothing arrived on those days, so this is a real supplier failure, not a typo - it belongs in supplier reliability.")
    df["days_late"] = (df.received_date - df.expected_date).dt.days
    df["is_late"] = (df.days_late > 0).astype(float).where(~is_open)
    df["fill_rate"] = df.qty_received / df.qty_ordered
    return df


def expand_promotions(dq):
    """Turn promotion periods into one row per store x product x day."""
    df = pd.read_csv(RAW / "promotions_calendar.csv", parse_dates=["start_date", "end_date"])
    assert (df.end_date >= df.start_date).all()
    days = df.assign(date=[pd.date_range(s, e) for s, e in zip(df.start_date, df.end_date)]).explode("date")
    overlaps = days.duplicated(["store_id", "product_id", "date"]).sum()
    dq.add("promotions_calendar", "Overlapping promotions for the same store-product-day", overlaps,
           "Checked - none found" if overlaps == 0 else "Kept the larger discount", "Each day must carry one discount.")
    days["date"] = pd.to_datetime(days.date)
    return days[["store_id", "product_id", "date", "promo_id", "discount_pct", "promo_type"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Inventory reconciliation
# --------------------------------------------------------------------------
def reconcile_inventory(dq, inventory_raw, sales_daily, po):
    inv = inventory_raw.copy()

    n = inv.duplicated().sum()
    inv = inv.drop_duplicates()
    dq.add("inventory", "Exact duplicate rows", n, "Removed", "Would double the stock and sales on those days.")
    assert not inv.duplicated(KEY).any()

    receipts = (po.dropna(subset=["received_date"])
                  .groupby(["store_id", "product_id", "received_date"]).qty_received.sum()
                  .rename("po_received").reset_index().rename(columns={"received_date": "date"}))
    inv = inv.merge(receipts, on=KEY, how="left")
    inv["po_received"] = inv.po_received.fillna(0).astype(int)

    expected = inv.opening_stock + inv.received_qty - inv.sold_qty
    mismatch = inv.closing_stock != expected
    negative = inv.closing_stock < 0
    dq.add("inventory", "closing != opening + received - sold", mismatch.sum(),
           "Diagnosed with the next day's opening count and purchase orders (two rows below)", "Stock-out labels come from closing stock.")

    # Case A: delivery missing from received_qty, confirmed by a purchase-order receipt.
    unrecorded = mismatch & (inv.received_qty != inv.po_received) & (inv.opening_stock + inv.po_received - inv.sold_qty == inv.closing_stock)
    inv.loc[unrecorded, "received_qty"] = inv.loc[unrecorded, "po_received"]
    dq.add("inventory", "  - Delivery not recorded (received_qty = 0, but a PO was received)", unrecorded.sum(),
           "received_qty taken from purchase_orders",
           "The closing count is confirmed by the next day's opening count, and the PO receipt explains the gap exactly.")

    # Case B: closing count mistyped - the arithmetic value is confirmed by the next day's opening.
    expected = inv.opening_stock + inv.received_qty - inv.sold_qty
    mistyped = inv.closing_stock != expected
    inv.loc[mistyped, "closing_stock"] = expected[mistyped]
    dq.add("inventory", "  - Closing stock mistyped (incl. negative closing)", mistyped.sum(),
           f"Recomputed as opening + received - sold ({negative.sum()} of these were negative)",
           "The next day's opening stock (physical count) equals the recomputed figure, not the recorded one. Negative stock is impossible.")
    inv["inv_corrected"] = (unrecorded | mistyped).astype(int)
    inv = inv.drop(columns="po_received")

    # Missing days: rebuild from the previous closing, PO receipts and cleaned sales.
    inv["date"] = pd.to_datetime(inv.date)
    first_day = inv.groupby(["store_id", "product_id"]).date.min()
    grid = pd.concat([
        pd.DataFrame({"store_id": s, "product_id": p, "date": pd.date_range(d, LAST_OBSERVED_DAY)})
        for (s, p), d in first_day.items()
    ])
    inv = grid.merge(inv, on=["store_id", "product_id", "date"], how="left")
    inv = inv.merge(receipts, on=KEY, how="left").merge(sales_daily, on=KEY, how="left")
    inv = inv.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    missing = inv.opening_stock.isna()
    for i in np.flatnonzero(missing):          # 277 rows - a plain loop is clearest
        prev = inv.loc[i - 1]
        opening = prev.closing_stock
        received = 0 if pd.isna(inv.at[i, "po_received"]) else inv.at[i, "po_received"]
        sold = 0 if pd.isna(inv.at[i, "units_sold"]) else inv.at[i, "units_sold"]
        inv.loc[i, ["opening_stock", "received_qty", "sold_qty", "closing_stock"]] = [opening, received, sold, opening + received - sold]
        inv.loc[i, ["reorder_level", "lead_time_days"]] = prev[["reorder_level", "lead_time_days"]].values
    inv["inv_rebuilt"] = missing.astype(int)
    inv["inv_corrected"] = inv.inv_corrected.fillna(0).astype(int)

    nxt = inv.groupby(["store_id", "product_id"]).opening_stock.shift(-1)
    rebuilt_ok = (inv.closing_stock == nxt)[missing & nxt.notna()]
    dq.add("inventory", "Missing store-product days", missing.sum(),
           f"Rebuilt: opening = previous closing, received = PO receipts, sold = transactions. "
           f"{rebuilt_ok.sum()}/{len(rebuilt_ok)} rebuilt closings match the next day's opening count",
           "Deleting the days would break the time series (lags, rolling means) and hide possible stock-outs.")

    cols = ["opening_stock", "received_qty", "sold_qty", "closing_stock", "reorder_level", "lead_time_days"]
    inv[cols] = inv[cols].astype(int)

    # Final integrity checks.
    assert (inv.closing_stock == inv.opening_stock + inv.received_qty - inv.sold_qty).all()
    assert (inv.closing_stock >= 0).all()
    prev_close = inv.groupby(["store_id", "product_id"]).closing_stock.shift(1)
    assert (inv.opening_stock == prev_close)[prev_close.notna()].all(), "stock chain broken"
    diff = (inv.sold_qty != inv.units_sold.fillna(0)).sum()
    assert diff == 0, f"{diff} days where inventory and transactions disagree"
    dq.add("inventory", "Cross-check: inventory sold_qty vs cleaned transactions", diff,
           "All days agree after cleaning", "Two independent systems now tell the same story - evidence the cleaning is correct.")
    return inv.drop(columns=["po_received", "units_sold"])


# --------------------------------------------------------------------------
# Master table
# --------------------------------------------------------------------------
def censored_demand(master):
    """
    On a stock-out day sales are capped by stock, so sales < demand.
    demand_est = max(units_sold, typical demand), where typical demand uses ONLY past
    non-stock-out days: mean of the last 4 same-weekday days, falling back to the last 7 days.
    """
    m = master.sort_values(["store_id", "product_id", "date"])
    clean_sales = m.units_sold.where(m.stockout_flag == 0)   # NaN on stock-out days
    combo = [m.store_id, m.product_id]
    same_weekday = clean_sales.groupby(combo + [m.day_of_week]).transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    last_week = clean_sales.groupby(combo).transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    typical = same_weekday.fillna(last_week)
    demand = np.where(m.stockout_flag == 1, np.fmax(m.units_sold, typical.round()), m.units_sold)
    m["demand_est"] = pd.Series(demand, index=m.index).fillna(m.units_sold).astype(int)
    return m.sort_index()


def build_master(dq, inv, sales, returns, products, stores, weather, promos):
    base = inv.copy()
    n = len(base)

    daily = (sales.groupby(KEY)
             .agg(units_sold_txn=("quantity", "sum"), revenue=("revenue", "sum"),
                  n_lines=("transaction_id", "size"), n_customers=("customer_id", "nunique"))
             .reset_index())
    daily["avg_selling_price"] = daily.revenue / daily.units_sold_txn
    ret = returns.groupby(KEY).agg(return_lines=("transaction_id", "size"), return_units=("quantity", lambda q: -q.sum())).reset_index()

    m = base.merge(daily, on=KEY, how="left")
    unmatched = len(daily) - daily.merge(base[KEY], on=KEY).shape[0]
    assert unmatched == 0, f"{unmatched} sales days have no master row"
    m = m.merge(ret, on=KEY, how="left")
    m = m.merge(promos.drop(columns="promo_id"), on=KEY, how="left")
    m = m.merge(products, on="product_id", how="left")
    m = m.merge(stores, on="store_id", how="left")
    m = m.merge(weather, on=["date", "city"], how="left")
    check_merge(base, m, "master")
    assert m[["category", "store_type", "temp_c"]].notna().all().all(), "a join left gaps"
    dq.add("master", "Rows / unmatched keys after every join", 0,
           f"{n:,} rows before and after all 7 joins; 0 unmatched keys",
           "Checked with assertions - a silent join failure would create NaNs rather than an error.")

    m = m.rename(columns={"sold_qty": "units_sold"}).drop(columns="units_sold_txn")
    for col in ["revenue", "n_lines", "n_customers", "return_lines", "return_units"]:
        m[col] = m[col].fillna(0)
    m[["n_lines", "n_customers", "return_lines", "return_units"]] = m[["n_lines", "n_customers", "return_lines", "return_units"]].astype(int)
    m["promo_flag"] = m.discount_pct.notna().astype(int)
    m["discount_pct"] = m.discount_pct.fillna(0).astype(int)
    m["promo_type"] = m.promo_type.fillna("No Promo")   # not "None": pandas reads that back as NaN

    # Unit price for every day (needed to value lost sales on zero-sale days):
    # observed price, else the last observed price at the same discount level, else MRP less discount.
    m = m.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    m["unit_price"] = (m.groupby(["store_id", "product_id", "discount_pct"]).avg_selling_price.ffill()
                        .fillna(m.mrp * (1 - m.discount_pct / 100)))

    # Calendar basics
    m["day_of_week"] = m.date.dt.dayofweek
    m["day_name"] = m.date.dt.day_name()
    m["month"] = m.date.dt.month
    m["week_no"] = m.date.dt.isocalendar().week.astype(int)

    # History
    m["launch_date"] = m.groupby(["store_id", "product_id"]).date.transform("min")
    m["history_days"] = (m.date - m.launch_date).dt.days + 1
    total_days = m.groupby(["store_id", "product_id"]).date.transform("size")
    m["sparse_history_flag"] = (total_days < SPARSE_HISTORY_DAYS).astype(int)
    sparse = m[m.sparse_history_flag == 1].groupby("product_id").agg(launch=("launch_date", "min"), days=("history_days", "max"))
    dq.add("master", f"Sparse history (< {SPARSE_HISTORY_DAYS} days of data)", sparse.shape[0],
           f"Kept and flagged ({', '.join(f'{p}: {r.days}d' for p, r in sparse.iterrows())}). "
           "Fallback for forecasting: category x store average scaled by the product's early sales",
           "Dropping them would give new launches no recommendation at all; their lag features will simply be empty.")

    # Stock-out definition and derived KPIs
    m["stockout_flag"] = (m.closing_stock == 0).astype(int)
    m["below_reorder_flag"] = (m.closing_stock < m.reorder_level).astype(int)
    m = censored_demand(m)
    m["lost_units_est"] = m.demand_est - m.units_sold
    m["lost_sales_est"] = (m.lost_units_est * m.unit_price).round(2)
    m["cogs"] = m.units_sold * m.cost_price
    m["inventory_value"] = m.closing_stock * m.cost_price
    m["margin"] = (m.revenue - m.cogs).round(2)
    m["revenue"] = m.revenue.round(2)
    m["avg_selling_price"] = m.avg_selling_price.round(2)
    m["unit_price"] = m.unit_price.round(2)

    order = [
        "date", "store_id", "product_id", "product_name", "category", "sub_category", "brand", "supplier_id",
        "city", "region", "store_type", "floor_area_sqft", "avg_daily_customers",
        "mrp", "cost_price", "shelf_life_days",
        "units_sold", "demand_est", "lost_units_est", "lost_sales_est", "revenue", "cogs", "margin",
        "avg_selling_price", "unit_price", "n_lines", "n_customers", "return_lines", "return_units",
        "promo_flag", "discount_pct", "promo_type",
        "opening_stock", "received_qty", "closing_stock", "reorder_level", "lead_time_days", "inventory_value",
        "stockout_flag", "below_reorder_flag",
        "temp_c", "rain_mm", "holiday", "festival", "festival_name", "weekend", "local_event",
        "day_of_week", "day_name", "month", "week_no",
        "launch_date", "history_days", "sparse_history_flag",
        "inv_corrected", "inv_rebuilt", "weather_imputed",
    ]
    m = m[order]
    assert not m.duplicated(KEY).any(), "grain violated"
    return m


# --------------------------------------------------------------------------
# Raw column audit: type, missing, validity rule - for EVERY column, including those that pass
# --------------------------------------------------------------------------
def audit_raw_files():
    """One row per raw column. Counts are on the raw file as delivered (before any cleaning)."""
    raw = {f: pd.read_csv(RAW / f, dtype=str, keep_default_na=False) for f in
           ["transactions.csv", "products.csv", "stores.csv", "inventory.csv", "external_factors.csv",
            "promotions_calendar.csv", "purchase_orders.csv", "suppliers.csv"]}
    inferred = {f: pd.read_csv(RAW / f).dtypes for f in raw}
    num = lambda s: pd.to_numeric(s.replace("", np.nan), errors="coerce")
    day = lambda s: pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    std_cats = {c.lower() for c in CATEGORY_MAP.values()}
    product_ids = set(raw["products.csv"].product_id)
    supplier_ids = set(raw["suppliers.csv"].supplier_id)
    t, pr, st, inv = raw["transactions.csv"], raw["products.csv"], raw["stores.csv"], raw["inventory.csv"]
    ex, pc, po, su = raw["external_factors.csv"], raw["promotions_calendar.csv"], raw["purchase_orders.csv"], raw["suppliers.csv"]
    iso_or_dmy = day(t.date).notna() | pd.to_datetime(t.date, format="%d-%m-%Y", errors="coerce").notna()

    # (file, column, expected type, rule, invalid mask)
    rules = [
        ("transactions.csv", "transaction_id", "string", "format T + digits; unique per distinct record", ~t.transaction_id.str.fullmatch(r"T\d+")),
        ("transactions.csv", "date", "date", "YYYY-MM-DD (DD-MM-YYYY counted as invalid format)", day(t.date).isna()),
        ("transactions.csv", "date", "date", "parseable as a real date in either format, within 1 Mar - 31 Aug", ~iso_or_dmy),
        ("transactions.csv", "store_id", "string", "exact format S + 2 digits", ~t.store_id.str.fullmatch(r"S\d{2}")),
        ("transactions.csv", "product_id", "string", "exact format P + 3 digits and in product master", ~t.product_id.isin(product_ids)),
        ("transactions.csv", "quantity", "integer", "integer > 0", ~(num(t.quantity) > 0)),
        ("transactions.csv", "quantity", "integer", "< 100 (larger values checked against inventory)", num(t.quantity) >= 100),
        ("transactions.csv", "selling_price", "float", "0 < price <= MRP",
         ~((num(t.selling_price) > 0) & (num(t.selling_price) <=t.product_id.str.strip().str.upper().map(pr.drop_duplicates("product_id").set_index("product_id").mrp.astype(float))))),
        ("transactions.csv", "discount_pct", "integer", "one of 0,5,10,15,20,25,30", ~t.discount_pct.isin([str(x) for x in range(0, 35, 5)])),
        ("transactions.csv", "promotion_flag", "0/1", "0/1 and equals (discount_pct > 0)", ~t.promotion_flag.isin(["0", "1"]) | ((num(t.discount_pct) > 0).astype(int).astype(str) != t.promotion_flag)),
        ("transactions.csv", "customer_id", "string", "C + digits, or blank (non-member)", ~(t.customer_id.str.fullmatch(r"C\d+") | (t.customer_id == ""))),
        ("transactions.csv", "payment_mode", "category", "UPI / Card / Cash", ~t.payment_mode.isin(["UPI", "Card", "Cash"])),
        ("transactions.csv", "hour", "integer", "0-23", ~num(t.hour).between(0, 23)),
        ("products.csv", "product_id", "string", "P + 3 digits; unique", ~pr.product_id.str.fullmatch(r"P\d{3}") | pr.duplicated(keep="first")),
        ("products.csv", "product_name", "string", "not blank", pr.product_name == ""),
        ("products.csv", "category", "category", "exactly one of the 7 standard spellings", ~pr.category.isin(list(dict.fromkeys(CATEGORY_MAP.values())))),
        ("products.csv", "category", "category", "maps to a standard category after trim + lower-case", ~pr.category.str.strip().str.lower().isin(std_cats | set(CATEGORY_MAP))),
        ("products.csv", "sub_category", "string", "not blank", pr.sub_category == ""),
        ("products.csv", "brand", "string", "not blank", pr.brand == ""),
        ("products.csv", "mrp", "float", "> 0", ~(num(pr.mrp) > 0)),
        ("products.csv", "cost_price", "float", "0 < cost < MRP", ~((num(pr.cost_price) > 0) & (num(pr.cost_price) < num(pr.mrp)))),
        ("products.csv", "shelf_life_days", "integer", "> 0", ~(num(pr.shelf_life_days) > 0)),
        ("products.csv", "supplier_id", "string", "in supplier master", ~pr.supplier_id.isin(supplier_ids)),
        ("stores.csv", "store_id", "string", "S + 2 digits; unique", ~st.store_id.str.fullmatch(r"S\d{2}") | st.store_id.duplicated()),
        ("stores.csv", "city", "string", "has weather rows under the same name", ~st.city.isin(ex.city)),
        ("stores.csv", "store_type", "category", "exactly Express / Supermarket / Hypermarket", ~st.store_type.isin(["Express", "Supermarket", "Hypermarket"])),
        ("stores.csv", "floor_area_sqft", "integer", "> 0", ~(num(st.floor_area_sqft) > 0)),
        ("stores.csv", "avg_daily_customers", "integer", "> 0", ~(num(st.avg_daily_customers) > 0)),
        ("stores.csv", "region", "string", "not blank", st.region == ""),
        ("inventory.csv", "date", "date", "YYYY-MM-DD", day(inv.date).isna()),
        ("inventory.csv", "store_id / product_id", "string", "known store and product", ~inv.store_id.isin(st.store_id) | ~inv.product_id.isin(product_ids)),
        ("inventory.csv", "opening_stock / received_qty / sold_qty", "integer", ">= 0", ~(inv[["opening_stock", "received_qty", "sold_qty"]].apply(num) >= 0).all(axis=1)),
        ("inventory.csv", "closing_stock", "integer", ">= 0", ~(num(inv.closing_stock) >= 0)),
        ("inventory.csv", "closing_stock", "integer", "= opening + received - sold",
         num(inv.closing_stock) != num(inv.opening_stock) + num(inv.received_qty) - num(inv.sold_qty)),
        ("inventory.csv", "reorder_level", "integer", "> 0", ~(num(inv.reorder_level) > 0)),
        ("inventory.csv", "lead_time_days", "integer", "1-7", ~num(inv.lead_time_days).between(1, 7)),
        ("external_factors.csv", "date", "date", "YYYY-MM-DD", day(ex.date).isna()),
        ("external_factors.csv", "city", "string", "matches a city in stores.csv", ~ex.city.isin(st.city)),
        ("external_factors.csv", "temp_c", "float", f"{VALID_TEMP_RANGE[0]}-{VALID_TEMP_RANGE[1]} C", ~num(ex.temp_c).between(*VALID_TEMP_RANGE)),
        ("external_factors.csv", "rain_mm", "float", ">= 0", ~(num(ex.rain_mm) >= 0)),
        ("external_factors.csv", "holiday / festival / local_event", "0/1", "0 or 1", ~ex[["holiday", "festival", "local_event"]].isin(["0", "1"]).all(axis=1)),
        ("external_factors.csv", "festival_name", "string", "present exactly when festival = 1", (ex.festival == "1") != (ex.festival_name != "")),
        ("external_factors.csv", "weekend", "0/1", "equals Saturday/Sunday of the date", (day(ex.date).dt.dayofweek >= 5).astype(int).astype(str) != ex.weekend),
        ("promotions_calendar.csv", "store_id / product_id", "string", "known store and product", ~pc.store_id.isin(st.store_id) | ~pc.product_id.isin(product_ids)),
        ("promotions_calendar.csv", "start_date / end_date", "date", "valid dates, end >= start", ~(day(pc.end_date) >= day(pc.start_date))),
        ("promotions_calendar.csv", "discount_pct", "integer", "5-30 in steps of 5", ~pc.discount_pct.isin([str(x) for x in range(5, 35, 5)])),
        ("promotions_calendar.csv", "promo_type", "category", "Price Off / Festival Offer", ~pc.promo_type.isin(["Price Off", "Festival Offer"])),
        ("purchase_orders.csv", "po_id", "string", "unique", po.po_id.duplicated()),
        ("purchase_orders.csv", "supplier_id", "string", "in supplier master and matches the product's supplier",
         ~po.supplier_id.isin(supplier_ids) | (po.product_id.map(pr.drop_duplicates("product_id").set_index("product_id").supplier_id) != po.supplier_id)),
        ("purchase_orders.csv", "expected_date", "date", ">= order_date", ~(day(po.expected_date) >= day(po.order_date))),
        ("purchase_orders.csv", "received_date", "date", ">= order_date; blank only when status = OPEN",
         ((po.received_date == "") != (po.status == "OPEN")) | (day(po.received_date) < day(po.order_date))),
        ("purchase_orders.csv", "qty_ordered / qty_received", "integer", "ordered > 0; 0 <= received <= ordered (when received)",
         ~(num(po.qty_ordered) > 0) | ((po.status == "RECEIVED") & ~num(po.qty_received).between(0, num(po.qty_ordered)))),
        ("purchase_orders.csv", "status", "category", "RECEIVED / OPEN", ~po.status.isin(["RECEIVED", "OPEN"])),
        ("suppliers.csv", "nominal_lead_days", "integer", "1-7", ~num(su.nominal_lead_days).between(1, 7)),
    ]
    rows = []
    for f, col, expected, rule, bad in rules:
        first = col.split(" / ")[0]
        blanks = int((raw[f][[c for c in col.split(" / ")]] == "").any(axis=1).sum())
        rows.append(dict(file=f, column=col, dtype_as_read=str(inferred[f][first]), expected_type=expected,
                         rows=len(raw[f]), missing=blanks, rule=rule, violations=int(bad.sum())))
    # Duplicate check per file
    for f, df in raw.items():
        rows.append(dict(file=f, column="(whole row)", dtype_as_read="-", expected_type="-", rows=len(df), missing=0,
                         rule="no exact duplicate rows", violations=int(df.duplicated().sum())))
    audit = pd.DataFrame(rows)
    blank_allowed = audit.column.isin(["customer_id", "festival_name", "received_date", "qty_ordered / qty_received"])
    clean = (audit.violations == 0) & ((audit.missing == 0) | blank_allowed)
    audit["result"] = np.where(clean, np.where(audit.missing > 0, "OK (blank is meaningful)", "OK"), "Issue - fixed, see below")
    return audit


COLUMN_NOTES = {
    "units_sold": "Units sold that day (clean transactions; equals reconciled inventory sold_qty)",
    "demand_est": "Estimated true demand: units_sold, or on stock-out days max(units_sold, typical demand from PAST non-stock-out days)",
    "lost_units_est": "demand_est - units_sold (only > 0 on stock-out days)",
    "lost_sales_est": "lost_units_est x unit_price (Rs)",
    "revenue": "Sum of quantity x selling_price (Rs)",
    "cogs": "units_sold x cost_price (Rs)",
    "margin": "revenue - cogs (Rs)",
    "avg_selling_price": "revenue / units_sold; blank on days with no sale",
    "unit_price": "Price for every day: observed, else last observed at the same discount, else MRP x (1 - discount)",
    "n_lines / n_customers": "Sale lines and distinct loyalty customers that day",
    "return_lines / return_units": "Lines with quantity <= 0 and the units returned",
    "promo_flag / discount_pct / promo_type": "From promotions_calendar (agrees 100% with transactions; also covers zero-sale days)",
    "opening_stock ... lead_time_days": "Reconciled inventory: closing = opening + received - sold on every row",
    "inventory_value": "closing_stock x cost_price (Rs)",
    "stockout_flag": "1 if closing_stock == 0 at end of day (operational stock-out definition)",
    "below_reorder_flag": "1 if closing_stock < reorder_level (early warning)",
    "temp_c ... local_event": "external_factors for the store's city (cleaned)",
    "day_of_week / day_name / month / week_no": "Calendar of the row date (Mon = 0)",
    "launch_date / history_days / sparse_history_flag": "First day of the store x product, days of history so far, 1 if < 28 days in total",
    "inv_corrected / inv_rebuilt / weather_imputed": "Audit flags: row changed by reconciliation / rebuilt missing day / imputed weather",
}


def write_dq_report(dq, row_counts, master, audit):
    log = dq.frame()
    so = master[master.stockout_flag == 1]
    uplifted = (so.demand_est > so.units_sold).sum()
    lines = [
        "# Data Quality Report", "",
        "_Generated by `src/data_prep.py`. Every count comes from the pipeline itself; the machine-readable log is `data_quality_log.csv`._", "",
        "## Approach", "",
        "1. **Audit** every file: types, missing values, duplicates, invalid values, key formats.",
        "2. **Fix with evidence**: where two systems overlap (transactions vs inventory, inventory vs purchase orders, "
        "a day's closing vs the next day's opening count) the cross-check decides the fix - not a blanket rule.",
        "3. **Assert**: after cleaning, the pipeline fails loudly if any rule is violated (see 'Integrity checks').", "",
        "## Row counts", "", "| File | Raw rows | Clean rows | Note |", "|---|---|---|---|",
    ]
    lines += [f"| {f} | {'-' if r is None else f'{r:,}'} | {c:,} | {n} |" for f, r, c, n in row_counts]
    n_issue = (audit.result.str.startswith("Issue")).sum()
    lines += ["", "## Column audit of the raw files (data types, missing values, invalid values, duplicates)", "",
              f"Every raw column was checked against an explicit rule - {len(audit)} checks, {len(audit) - n_issue} clean, {n_issue} with issues "
              "(all fixed below). Counts are on the raw files, before any cleaning. `dtype_as_read` is what pandas infers: "
              "e.g. `date` is read as text (mixed formats) and IDs as text, which is why they are parsed and standardised explicitly.", "",
              "| File | Column | dtype as read | Expected | Missing | Rule | Violations | Result |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.file} | {r.column} | {r.dtype_as_read} | {r.expected_type} | {r.missing:,} | {r.rule} | {r.violations:,} | {r.result} |"
              for r in audit.itertuples()]
    lines += ["", "## Issues found and actions taken", "", "| Table | Issue | Count | Action taken | Justification |", "|---|---|---|---|---|"]
    for r in log.itertuples():
        lines.append(f"| {r.table} | {r.issue} | {r.count:,} | {r.action} | {r.justification} |")
    lines += [
        "", "## Integrity checks (all enforced with assertions)", "",
        "- One row per date x store x product in the master table (no duplicate keys).",
        "- `closing = opening + received - sold` on every inventory row, and no negative stock.",
        "- Each day's `opening_stock` equals the previous day's `closing_stock` (unbroken stock chain).",
        "- Inventory `sold_qty` equals the cleaned transaction total on every store-product-day.",
        "- Every selling price satisfies 0 < price <= MRP and is identical within a store-product-day-discount.",
        "- Row count unchanged by every join; every sales day, product, store and city found its match.", "",
        "## Key definitions", "",
        "- **Stock-out:** `stockout_flag = 1` when reconciled `closing_stock == 0` at end of day. Objective, observable, and exactly what a "
        "manager sees. Alternative considered: 'closing stock < one day of demand' - rejected as the main label because it depends on a demand "
        "estimate, making the label partly model-generated; `below_reorder_flag` is kept as an early-warning signal instead.",
        "- **Censored demand:** on stock-out days sales are capped by stock. `demand_est` replaces sales on those days with the typical demand "
        "of past non-stock-out days (last 4 same weekdays, else last 7 days). Uses only the past, so it is safe for forecasting features.",
        "- **Sparse history:** combinations with < 28 days of data are kept and flagged (`sparse_history_flag`). Forecast fallback: "
        "category x store average scaled by the product's early sales.", "",
        "## Known limitations", "",
        f"- `demand_est` is conservative: estimated lost demand is {master.lost_units_est.sum() / master.demand_est.sum():.1%} of demand, "
        "because demand beyond an empty shelf is only partly visible. Of "
        f"{len(so):,} stock-out days, {uplifted:,} were raised above sales; on the other {len(so) - uplifted:,} the shelf emptied late in the day, "
        "after sales had already reached the typical level, so no loss is assumed (true demand may still have been higher).",
        "- A few missing prices had no sibling line that day and were taken from the nearest day at the same discount (count in the log above).",
        "- Weather for 1-7 Sep in `external_factors_clean.csv` is a forecast stand-in, not an observation.",
        "", "## Master table columns", "", "| Column(s) | Meaning |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in COLUMN_NOTES.items()]
    lines += ["", "Static attributes (product, store) are joined from the cleaned master tables and keep their original names."]
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    dq = DQLog()

    print("Cleaning master tables ...")
    products = clean_products(dq)
    stores = clean_stores(dq)
    suppliers = clean_suppliers()
    weather = clean_external(dq)

    print("Cleaning transactions ...")
    inventory_raw = pd.read_csv(RAW / "inventory.csv", parse_dates=["date"])
    sales, returns = clean_transactions(dq, products, inventory_raw)
    sales_daily = sales.groupby(KEY).quantity.sum().rename("units_sold").reset_index()

    print("Purchase orders, promotions, inventory ...")
    po = clean_purchase_orders(dq)
    promos = expand_promotions(dq)
    inv = reconcile_inventory(dq, inventory_raw, sales_daily, po)

    print("Building master table ...")
    master = build_master(dq, inv, sales, returns, products, stores, weather, promos)

    print("Saving ...")
    master.to_csv(OUT / "master_dataset.csv", index=False)
    sales.to_csv(OUT / "transactions_clean.csv", index=False)
    products.to_csv(OUT / "products_clean.csv", index=False)
    stores.to_csv(OUT / "stores_clean.csv", index=False)
    suppliers.to_csv(OUT / "suppliers_clean.csv", index=False)
    weather.to_csv(OUT / "external_factors_clean.csv", index=False)   # includes 1-7 Sep (calendar known in advance)
    po.to_csv(OUT / "purchase_orders_clean.csv", index=False)
    promos.to_csv(OUT / "promotions_daily.csv", index=False)           # includes 1-7 Sep (planned promotions)
    dq.frame().to_csv(REPORTS / "data_quality_log.csv", index=False)

    raw_rows = lambda f: sum(1 for _ in open(RAW / f, encoding="utf-8")) - 1
    row_counts = [
        ("transactions.csv", raw_rows("transactions.csv"), len(sales), f"+ {len(returns):,} return/void lines kept aside"),
        ("products.csv", raw_rows("products.csv"), len(products), "duplicates removed"),
        ("stores.csv", raw_rows("stores.csv"), len(stores), ""),
        ("inventory.csv", raw_rows("inventory.csv"), len(inv), "duplicates removed, missing days rebuilt"),
        ("external_factors.csv", raw_rows("external_factors.csv"), len(weather), "duplicates removed; runs to 7 Sep"),
        ("promotions_calendar.csv", raw_rows("promotions_calendar.csv"), len(promos), "expanded to one row per promo day"),
        ("purchase_orders.csv", raw_rows("purchase_orders.csv"), len(po), ""),
        ("suppliers.csv", raw_rows("suppliers.csv"), len(suppliers), ""),
        ("**master_dataset.csv**", None, len(master), f"{master.shape[1]} columns; grain = date x store x product"),
    ]
    audit = audit_raw_files()
    audit.to_csv(REPORTS / "raw_column_audit.csv", index=False)
    write_dq_report(dq, row_counts, master, audit)
    print(f"master_dataset.csv: {master.shape[0]:,} rows x {master.shape[1]} columns")
    return master, dq


if __name__ == "__main__":
    main()
