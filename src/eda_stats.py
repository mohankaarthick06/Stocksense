"""
StockSense - Round 1 EDA, KPIs and statistical tests (Person 1: Data Analyst).

Every chart answers one business question. Every insight sentence is computed
from the data (no hand-typed numbers), so re-running keeps the reports honest.

Run from the repo root (after src/data_prep.py):
    python src/eda_stats.py
"""
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ALPHA = 0.05

# ---------------------------------------------------------------- chart style
# Categorical slots are assigned in a fixed order and follow the entity, never its rank.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DE_EMPHASIS = "#c3c2b7"
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

CATEGORIES = ["Beverages", "Dairy", "Frozen", "Groceries", "Household", "Personal Care", "Snacks"]
STORES = ["S01", "S02", "S03", "S04", "S05", "S06"]
FORMATS = ["Express", "Hypermarket", "Supermarket"]
CAT_COLOR = dict(zip(CATEGORIES, SLOTS))
STORE_COLOR = dict(zip(STORES, SLOTS))
FORMAT_COLOR = dict(zip(FORMATS, SLOTS))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.edgecolor": BASELINE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "lines.linewidth": 2, "legend.frameon": False, "legend.labelcolor": INK2, "axes.axisbelow": True,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def rupees(x):
    """Indian-style money: Rs 1.2 Cr / Rs 3.4 L / Rs 5,600."""
    if abs(x) >= 1e7:
        return f"Rs {x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x / 1e5:.1f} L"
    return f"Rs {x:,.0f}"


def fmt_p(p):
    return "< 1e-300" if p == 0 else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")


def md_table(df):
    """DataFrame -> markdown table (avoids an extra dependency on `tabulate`)."""
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return fmt_p(v) if abs(v) < 1e-3 and v != 0 else f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def suptitle(fig, question):
    fig.suptitle(question, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)


def load_master():
    m = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date", "launch_date"])
    m["festival_name"] = m.festival_name.fillna("")
    return m


# ======================================================================= KPIs
def kpis(m):
    """The brief's KPI list, computed overall and per store."""
    def one(g):
        days = g.date.nunique()
        promo = g[g.promo_flag == 1].units_sold.mean()
        normal = g[g.promo_flag == 0].units_sold.mean()
        last = g[g.date == g.date.max()]
        recent_daily_demand = g[g.date > g.date.max() - pd.Timedelta(days=28)].groupby("date").demand_est.sum().mean()
        avg_inv_value = g.groupby("date").inventory_value.sum().mean()
        return pd.Series({
            "Revenue (Rs)": g.revenue.sum(),
            "Units Sold": g.units_sold.sum(),
            "Gross Margin (Rs)": g.margin.sum(),
            "Stock-out Rate": g.stockout_flag.mean(),
            "Inventory Turnover (period)": g.cogs.sum() / avg_inv_value,
            "Inventory Turnover (annualised)": g.cogs.sum() / avg_inv_value * 365 / days,
            "Days of Inventory (31 Aug)": last.closing_stock.sum() / recent_daily_demand,
            "Inventory Value at Cost, 31 Aug (Rs)": last.inventory_value.sum(),
            "Promotion Lift": (promo - normal) / normal,
            "Estimated Lost Sales (Rs)": g.lost_sales_est.sum(),
            "Lost Sales as % of Revenue": g.lost_sales_est.sum() / g.revenue.sum(),
            "Returned Units": g.return_units.sum(),
        })

    overall = one(m).rename("All stores")
    by_store = m.groupby("store_id").apply(one, include_groups=False)
    table = pd.concat([overall.to_frame().T, by_store])
    notes = {
        "Revenue (Rs)": "Sum(quantity x selling price) over clean sale lines",
        "Units Sold": "Sum of quantity (returns excluded)",
        "Gross Margin (Rs)": "Revenue - units x cost_price",
        "Stock-out Rate": "Store-product-days with closing stock = 0 / all store-product-days",
        "Inventory Turnover (period)": "COGS / average daily inventory value at cost, over 1 Mar - 31 Aug",
        "Inventory Turnover (annualised)": "Period turnover x 365 / 184",
        "Days of Inventory (31 Aug)": "Closing stock on 31 Aug / average daily demand of the last 28 days",
        "Inventory Value at Cost, 31 Aug (Rs)": "Closing stock x cost_price on the last day",
        "Promotion Lift": "(mean daily units on promo days - on non-promo days) / non-promo (pooled; see test T1 for the paired version)",
        "Estimated Lost Sales (Rs)": "(demand_est - units_sold) x unit price on stock-out days",
        "Lost Sales as % of Revenue": "Estimated lost sales / revenue",
        "Returned Units": "Units on quantity <= 0 lines",
    }
    return table, notes


def monthly_trend(m):
    t = m.groupby(m.date.dt.to_period("M")).agg(
        revenue=("revenue", "sum"), units=("units_sold", "sum"),
        stockout_rate=("stockout_flag", "mean"), lost_sales=("lost_sales_est", "sum"))
    t["revenue_growth_mom"] = t.revenue.pct_change()
    return t


# ======================================================================= EDA
def eda_category_revenue(m):
    q = "Which categories generate most revenue?"
    cat = m.groupby("category").revenue.sum().sort_values()
    share = cat / cat.sum()
    prod = m.groupby(["product_id", "product_name"]).revenue.sum().sort_values(ascending=False).reset_index()
    prod["cum_share"] = prod.revenue.cumsum() / prod.revenue.sum()
    n80 = int((prod.cum_share < 0.80).sum() + 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.barh(cat.index, cat / 1e5, color=[CAT_COLOR[c] for c in cat.index], height=0.6)
    for y, (v, s) in enumerate(zip(cat / 1e5, share)):
        a1.text(v, y, f"  {v:,.0f} L  ({s:.0%})", va="center", color=INK2, fontsize=9)
    a1.set_xlabel("Revenue (Rs lakh)")
    a1.set_title("Revenue by category")
    a1.grid(axis="y", visible=False); a1.grid(axis="x")
    a1.set_xlim(0, cat.max() / 1e5 * 1.35)

    x = np.arange(1, len(prod) + 1)
    a2.plot(x, prod.cum_share, color=SLOTS[0])
    a2.axhline(0.8, color=MUTED, lw=1, ls="--")
    a2.axvline(n80, color=MUTED, lw=1, ls="--")
    a2.scatter([n80], [prod.cum_share.iloc[n80 - 1]], s=40, color=SLOTS[0], zorder=3, edgecolor=SURFACE, lw=2)
    a2.annotate(f"{n80} of {len(prod)} products ({n80 / len(prod):.0%})\n= 80% of revenue",
                (n80, 0.8), xytext=(n80 + 6, 0.55), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    a2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    a2.set_xlabel("Products ranked by revenue"); a2.set_ylabel("Cumulative share of revenue")
    a2.set_title("Pareto: how concentrated is revenue?")
    a2.set_ylim(0, 1.02)
    suptitle(fig, q); fig.tight_layout()

    top = prod.head(5)
    top10 = prod.cum_share.iloc[9]
    spread = "is NOT a classic 80/20 pattern - it is spread widely" if n80 / len(prod) > 0.35 else "is concentrated"
    insight = (f"{cat.index[-1]} ({share.iloc[-1]:.0%}) and {cat.index[-2]} ({share.iloc[-2]:.0%}) bring in "
               f"{share.iloc[-1] + share.iloc[-2]:.0%} of revenue; {cat.index[0]} is smallest ({share.iloc[0]:.0%}). "
               f"Revenue {spread}: it takes {n80} of {len(prod)} products ({n80 / len(prod):.0%}) to reach 80%, "
               f"although the top 10 alone give {top10:.0%}. Top products: {', '.join(top.product_name)}. "
               f"Action: the top 10 deserve the tightest availability targets and daily stock checks, but a broad range "
               f"matters, so replenishment must work for the whole catalogue, not only a few hero SKUs.")
    table = cat.sort_values(ascending=False).to_frame("revenue").assign(share=share)
    return dict(name="01_category_revenue_pareto", question=q, fig=fig, insight=insight, table=table)


def eda_store_trend(m):
    q = "Which stores are growing or declining?"
    # Like-for-like: only combinations present from day one, full Monday-Sunday weeks.
    lfl = m[m.launch_date == m.date.min()]
    lfl = lfl[(lfl.date >= "2026-03-02") & (lfl.date <= "2026-08-30")]
    week = lfl.date - pd.to_timedelta(lfl.day_of_week, unit="D")
    weekly = lfl.groupby([week.rename("week"), "store_id"]).units_sold.sum().unstack()
    base = weekly.iloc[:4].mean()
    index = weekly / base * 100
    growth = (weekly.iloc[-4:].mean() / base - 1).sort_values()
    slope = weekly.apply(lambda s: np.polyfit(np.arange(len(s)), s / s.mean(), 1)[0])  # share of mean per week

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for s in STORES:
        a1.plot(index.index, index[s], color=STORE_COLOR[s], label=s, lw=1.8)
    a1.axhline(100, color=MUTED, lw=1, ls="--")
    a1.set_ylabel("Weekly units, index (first 4 weeks = 100)")
    a1.set_title("Weekly units sold, like-for-like products")
    a1.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    a1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))

    stores = m.drop_duplicates("store_id").set_index("store_id")
    labels = [f"{s} {stores.city[s]} ({stores.store_type[s]})" for s in growth.index]
    a2.barh(labels, growth * 100, color=[POSITIVE if g >= 0 else NEGATIVE for g in growth], height=0.6)
    a2.axvline(0, color=BASELINE, lw=1)
    for y, g in enumerate(growth):
        a2.text(g * 100, y, f" {g:+.0%} " if g >= 0 else f" {g:+.0%} ", va="center",
                ha="left" if g >= 0 else "right", color=INK2, fontsize=9)
    a2.set_xlabel("Change: last 4 weeks vs first 4 weeks (%)")
    a2.set_title("Growth by store")
    a2.grid(axis="y", visible=False); a2.grid(axis="x")
    lim = max(abs(growth).max() * 100 * 1.4, 10)
    a2.set_xlim(-lim, lim)
    suptitle(fig, q); fig.tight_layout()

    insight = (f"Like-for-like weekly units (last 4 weeks vs first 4): fastest growth {growth.index[-1]} "
               f"({growth.iloc[-1]:+.0%}), then {growth.index[-2]} ({growth.iloc[-2]:+.0%}); "
               f"weakest {growth.index[0]} ({growth.iloc[0]:+.0%}). "
               f"Action: reorder levels set on old sales under-stock growing stores and over-stock declining ones - "
               f"growth must be a forecasting feature, and {growth.index[0]}'s stock should be trimmed.")
    table = pd.DataFrame({"growth_last4_vs_first4": growth, "trend_per_week_pct_of_mean": slope[growth.index] * 100})
    return dict(name="02_store_growth", question=q, fig=fig, insight=insight, table=table)


def promo_pairs(m, value="units_sold", min_days=3):
    """Mean daily value on promo vs non-promo days for each store x product (within-product comparison)."""
    g = m[m.sparse_history_flag == 0]
    stats_ = g.groupby(["store_id", "product_id", "category", "promo_flag"])[value].agg(["mean", "size"]).unstack("promo_flag")
    stats_ = stats_[(stats_[("size", 1)] >= min_days) & (stats_[("size", 0)] >= min_days)]
    out = pd.DataFrame({"promo": stats_[("mean", 1)], "normal": stats_[("mean", 0)]}).reset_index()
    out["lift"] = out.promo / out.normal - 1
    return out[out.normal > 0]


def eda_promotions(m):
    q = "Do promotions increase units sold?"
    pairs = promo_pairs(m)
    by_cat = pairs.groupby("category").lift.median().sort_values()
    g = m[m.sparse_history_flag == 0]
    # Lift by discount depth, normalised within each store x product.
    base = g[g.promo_flag == 0].groupby(["store_id", "product_id"]).units_sold.mean().rename("base")
    d = g.join(base, on=["store_id", "product_id"])
    d = d[d.base > 0]
    by_disc = (d.units_sold / d.base).groupby(d.discount_pct).mean() - 1
    so = m.groupby("promo_flag").stockout_flag.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.barh(by_cat.index, by_cat * 100, color=[CAT_COLOR[c] for c in by_cat.index], height=0.6)
    for y, v in enumerate(by_cat):
        a1.text(v * 100, y, f"  {v:+.0%}", va="center", color=INK2, fontsize=9)
    a1.set_xlabel("Median lift in daily units on promo days (%)")
    a1.set_title("Promotion lift by category (same product, same store)")
    a1.grid(axis="y", visible=False); a1.grid(axis="x")
    a1.set_xlim(0, by_cat.max() * 100 * 1.3)

    a2.plot(by_disc.index, by_disc * 100, color=SLOTS[0], marker="o", ms=7, mec=SURFACE, mew=2)
    for x_, y_ in by_disc.items():
        a2.annotate(f"{y_:+.0%}", (x_, y_ * 100), xytext=(-12, 8), textcoords="offset points", ha="center", color=INK2, fontsize=9)
    a2.set_xlabel("Discount (%)"); a2.set_ylabel("Average lift vs the product's normal day (%)")
    a2.set_title("Deeper discounts sell more")
    a2.set_xticks(by_disc.index)
    suptitle(fig, q); fig.tight_layout()

    insight = (f"Yes. For the same product in the same store, promo days sell a median {pairs.lift.median():+.0%} more units "
               f"({(pairs.lift > 0).mean():.0%} of {len(pairs)} store-product pairs gain). "
               f"Lift is highest in {by_cat.index[-1]} ({by_cat.iloc[-1]:+.0%}) and lowest in {by_cat.index[0]} ({by_cat.iloc[0]:+.0%}); "
               f"it rises with discount depth ({by_disc.index[1]}% off: {by_disc.iloc[1]:+.0%}, {by_disc.index.max()}% off: {by_disc.iloc[-1]:+.0%}). "
               f"But the stock-out rate is {so[1]:.1%} on promo days vs {so[0]:.1%} otherwise - promotions are not being stocked for. "
               f"Action: raise the order before every planned promotion by the category's expected lift.")
    return dict(name="03_promotion_lift", question=q, fig=fig, insight=insight,
                table=pd.DataFrame({"median_lift": by_cat.sort_values(ascending=False)}))


def eda_weekend(m):
    q = "How does weekend behaviour differ?"
    g = m[m.sparse_history_flag == 0]
    daily = g.groupby(["store_type", "day_of_week"]).units_sold.mean().unstack(0)
    index = daily / daily.loc[0:4].mean() * 100
    uplift = daily.loc[5:6].mean() / daily.loc[0:4].mean() - 1
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for f in FORMATS:
        ax.plot(range(7), index[f], color=FORMAT_COLOR[f], marker="o", ms=7, mec=SURFACE, mew=2,
                label=f"{f} (weekend {uplift[f]:+.0%})")
        ax.text(6.15, index[f].iloc[-1], f, color=INK2, va="center", fontsize=9)
    ax.axvspan(4.5, 6.5, color=GRID, alpha=0.5, lw=0)
    ax.axhline(100, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(7), days); ax.set_xlim(-0.3, 7)
    ax.set_ylabel("Avg daily units per product, index (Mon-Fri = 100)")
    ax.set_title("Weekend demand vs weekdays, by store format")
    ax.legend(loc="upper left")
    suptitle(fig, q); fig.tight_layout()

    peak = {f: days[int(index[f].values.argmax())] for f in FORMATS}
    insight = (f"The weekend effect depends on the format: Hypermarket {uplift['Hypermarket']:+.0%}, "
               f"Supermarket {uplift['Supermarket']:+.0%}, Express {uplift['Express']:+.0%} vs weekdays "
               f"(peak day: {', '.join(f'{f} {d}' for f, d in peak.items())}). "
               f"Action: hypermarket/supermarket deliveries should land by Friday; a single weekly pattern for all stores would be wrong.")
    return dict(name="04_weekend_vs_weekday", question=q, fig=fig, insight=insight,
                table=pd.DataFrame({"weekend_uplift": uplift}))


def volatility_table(m):
    g = m[m.sparse_history_flag == 0].sort_values("date")
    rolling_std = g.groupby(["store_id", "product_id"]).demand_est.transform(lambda s: s.rolling(7).std())
    g = g.assign(rolling_std_7=rolling_std)
    v = g.groupby(["store_id", "product_id", "product_name", "category"]).agg(
        mean_daily=("demand_est", "mean"), std_daily=("demand_est", "std"), avg_rolling_std_7=("rolling_std_7", "mean"))
    v["cv"] = v.std_daily / v.mean_daily
    return v.reset_index()


def eda_volatility(m):
    q = "Which products are volatile?"
    v = volatility_table(m)
    by_cat = v.groupby("category").cv.median().sort_values()
    by_prod = v.groupby(["product_name", "category"]).cv.median().sort_values(ascending=False).head(12).iloc[::-1]
    # Slow movers have high CV from counting noise alone; flag products volatile *for their size*.
    v["expected_cv"] = 1 / np.sqrt(v.mean_daily)          # Poisson noise floor
    v["excess_cv"] = v.cv / v.expected_cv

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.barh(by_cat.index, by_cat, color=[CAT_COLOR[c] for c in by_cat.index], height=0.6)
    for y, c in enumerate(by_cat):
        a1.text(c, y, f"  {c:.2f}", va="center", color=INK2, fontsize=9)
    a1.set_xlabel("Median coefficient of variation (std / mean of daily demand)")
    a1.set_title("Volatility by category")
    a1.grid(axis="y", visible=False); a1.grid(axis="x")
    a1.set_xlim(0, by_cat.max() * 1.25)

    a2.barh([p for p, _ in by_prod.index], by_prod.values, color=[CAT_COLOR[c] for _, c in by_prod.index], height=0.6)
    for y, c in enumerate(by_prod.values):
        a2.text(c, y, f"  {c:.2f}", va="center", color=INK2, fontsize=9)
    a2.set_xlabel("Median CV across stores")
    a2.set_title("12 most volatile products")
    a2.grid(axis="y", visible=False); a2.grid(axis="x")
    a2.set_xlim(0, by_prod.max() * 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CATEGORIES if c in set(by_prod.index.get_level_values(1))]
    a2.legend(handles, [c for c in CATEGORIES if c in set(by_prod.index.get_level_values(1))], loc="lower right", fontsize=8.5)
    suptitle(fig, q); fig.tight_layout()

    corr = stats.spearmanr(v.mean_daily, v.cv)[0]
    excess = v.sort_values("excess_cv", ascending=False).head(3)
    insight = (f"{by_cat.index[-1]} is the most volatile category (median CV {by_cat.iloc[-1]:.2f}) and {by_cat.index[0]} the most stable "
               f"({by_cat.iloc[0]:.2f}). Most volatile products: {', '.join(p for p, _ in by_prod.index[::-1][:3])}. "
               f"Part of this is size: slow sellers look volatile from counting noise (Spearman rho between mean demand and CV = {corr:.2f}). "
               f"Volatile beyond what their size explains: "
               f"{', '.join(f'{r.product_name} at {r.store_id}' for r in excess.itertuples())}. "
               f"Action: safety stock must scale with each product's volatility, not a flat percentage.")
    return dict(name="05_demand_volatility", question=q, fig=fig, insight=insight,
                table=v.sort_values("cv", ascending=False).head(20))


def eda_stockout_heatmap(m):
    q = "Which stores repeatedly stock out?"
    rate = m.pivot_table(index="store_id", columns="category", values="stockout_flag", aggfunc="mean")
    store_rate = m.groupby("store_id").stockout_flag.mean()
    stores = m.drop_duplicates("store_id").set_index("store_id")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(rate.values, cmap=BLUES, aspect="auto", vmin=0)
    ax.set_xticks(range(rate.shape[1]), rate.columns)
    ax.set_yticks(range(rate.shape[0]), [f"{s}  {stores.city[s]} ({stores.store_type[s]}) - {store_rate[s]:.1%}" for s in rate.index])
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, rate.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, rate.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    vmax = np.nanmax(rate.values)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            val = rate.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9.5,
                        color="white" if val > vmax * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, format=mtick.PercentFormatter(1, decimals=0), fraction=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Share of days with closing stock = 0", color=INK2)
    ax.set_title("Stock-out rate by store and category (label: store overall rate)")
    suptitle(fig, q); fig.tight_layout()

    flat = rate.stack().sort_values(ascending=False)
    cat_rate = m.groupby("category").stockout_flag.mean().sort_values()
    insight = (f"Stock-outs are concentrated: worst store {store_rate.idxmax()} ({store_rate.max():.1%}) and "
               f"{store_rate.sort_values().index[-2]} ({store_rate.sort_values().iloc[-2]:.1%}) vs best {store_rate.idxmin()} ({store_rate.min():.1%}). "
               f"{cat_rate.index[-1]} stock out on {cat_rate.iloc[-1]:.0%} of days - {cat_rate.iloc[-1] / cat_rate.iloc[:-1].mean():.1f}x the other categories' average. "
               f"Worst cells: {', '.join(f'{s}-{c} {v:.0%}' for (s, c), v in flat.head(3).items())}. "
               f"Action: {cat_rate.index[-1]} needs a supply fix (see supplier chart), not just higher reorder levels.")
    return dict(name="06_stockout_heatmap", question=q, fig=fig, insight=insight, table=rate)


def eda_supplier(m, po):
    q = "Are some suppliers causing stock-outs?"
    rec = po[po.status == "RECEIVED"].copy()
    sup = rec.groupby("supplier_id").agg(late_rate=("is_late", "mean"), avg_days_late=("days_late", lambda d: d[d > 0].mean()),
                                         fill_rate=("fill_rate", "mean"), orders=("po_id", "size"))
    cats = m.groupby("supplier_id").category.agg(lambda c: ", ".join(sorted(c.unique())))
    so = m.groupby("supplier_id").stockout_flag.mean()
    sup = sup.join(cats.rename("categories")).join(so.rename("stockout_rate")).sort_values("late_rate")
    worst = sup.late_rate.idxmax()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = [SLOTS[1] if s == worst else DE_EMPHASIS for s in sup.index]
    a1.barh(sup.index, sup.late_rate * 100, color=colors, height=0.6)
    for y, (r, d) in enumerate(zip(sup.late_rate, sup.avg_days_late)):
        a1.text(r * 100, y, f"  {r:.0%}  (avg {d:.1f} d late)", va="center", color=INK2, fontsize=9)
    a1.set_xlabel("Purchase orders delivered late (%)")
    a1.set_title("Late-delivery rate by supplier")
    a1.grid(axis="y", visible=False); a1.grid(axis="x")
    a1.set_xlim(0, sup.late_rate.max() * 100 * 1.6)

    a2.scatter(sup.late_rate * 100, sup.stockout_rate * 100, s=60, color=colors, edgecolor=SURFACE, lw=2, zorder=3)
    a2.annotate(f"{worst} ({sup.categories[worst]})", (sup.late_rate[worst] * 100, sup.stockout_rate[worst] * 100),
                xytext=(-8, -4), textcoords="offset points", ha="right", fontsize=9, color=INK2)
    rest = sup.drop(worst)
    a2.annotate("other 9 suppliers", (rest.late_rate.max() * 100, rest.stockout_rate.max() * 100),
                xytext=(10, 8), textcoords="offset points", fontsize=9, color=INK2)
    a2.set_xlabel("Late-delivery rate (%)"); a2.set_ylabel("Stock-out rate of its products (%)")
    a2.set_title("Late suppliers -> empty shelves")
    a2.grid(axis="x")
    suptitle(fig, q); fig.tight_layout()

    others = rec[rec.supplier_id != worst].is_late.mean()
    insight = (f"{worst} ({sup.categories[worst]}) delivers {sup.late_rate[worst]:.0%} of orders late vs {others:.0%} for all other suppliers, "
               f"and its products stock out on {sup.stockout_rate[worst]:.0%} of days vs {so.drop(worst).mean():.0%} average for the rest. "
               f"Action: plan {worst} orders on its real (longer) lead time, hold extra safety stock and escalate with the supplier.")
    return dict(name="07_supplier_reliability", question=q, fig=fig, insight=insight, table=sup)


def censoring_groups(m):
    """Combinations that stocked out in May (so their May sales under-stated demand) vs the rest."""
    may = m[m.date.dt.month == 5].groupby(["store_id", "product_id"]).stockout_flag.max().rename("censored_in_may")
    return m.join(may, on=["store_id", "product_id"]).dropna(subset=["censored_in_may"])


def recalibration_table(m):
    """The 1 June reset of reorder levels, per store x product, against May sales and May estimated demand."""
    k = ["store_id", "product_id"]
    may = m[m.date.dt.month == 5].groupby(k).agg(
        may_sales=("units_sold", "mean"), may_demand=("demand_est", "mean"),
        censored_in_may=("stockout_flag", "max"), level_before=("reorder_level", "last"))
    after = m[m.date == "2026-06-01"].set_index(k).reorder_level.rename("level_after")
    t = may.join(after).dropna()
    t["days_of_sales"] = t.level_after / t.may_sales
    t["days_of_demand"] = t.level_after / t.may_demand
    t["level_change"] = t.level_after / t.level_before - 1
    return t


def eda_censoring(m):
    q = "Why do the same products keep stocking out?"
    g = censoring_groups(m)
    g = g[g.date.dt.month != 5]       # May defines the groups, so it is left out of the comparison
    period = np.where(g.date < "2026-05-01", "Mar-Apr (before)", "Jun-Aug (after reset)")
    rates = g.groupby([period, "censored_in_may"]).stockout_flag.mean().unstack().sort_index(ascending=False)
    recal = recalibration_table(m)
    cover = recal.groupby("censored_in_may")[["days_of_sales", "days_of_demand"]].median()
    change = recal.groupby("censored_in_may").level_change.median()

    labels = {1.0: "Stocked out in May", 0.0: "No stock-out in May"}
    colors = {1.0: SLOTS[1], 0.0: SLOTS[0]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.36
    for ax, table, fmt, ylabel, title in [
        (a1, rates * 100, "{:.1f}%", "Stock-out rate (%)", "Stock-out rate before and after the 1 June reset"),
        (a2, cover.T, "{:.2f}", "New reorder level / May daily average (days)", "The reset used sales, not demand"),
    ]:
        for i, k in enumerate([1.0, 0.0]):
            vals = table[k] if ax is a1 else table[k]
            xs = np.arange(len(vals)) + (i - 0.5) * w
            ax.bar(xs, vals, width=w * 0.94, color=colors[k], label=labels[k])
            for x_, v in zip(xs, vals):
                ax.text(x_, v, fmt.format(v), ha="center", va="bottom", color=INK2, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, np.nanmax(table.values) * 1.3)
        ax.legend(loc="upper right", fontsize=8.5)
    a1.set_xticks(range(len(rates)), rates.index)
    a2.set_xticks(range(2), ["Days of May SALES", "Days of May estimated DEMAND"])
    suptitle(fig, q); fig.tight_layout()

    insight = (f"On 1 June every reorder level was reset to about {cover.loc[0.0, 'days_of_sales']:.1f} days of May *sales* - one rule for all. "
               f"For products that ran out in May, sales were capped by the empty shelf, so the new level covers only "
               f"{cover.loc[1.0, 'days_of_demand']:.2f} days of estimated demand vs {cover.loc[0.0, 'days_of_demand']:.2f} for the rest. "
               f"Their levels did rise (median {change[1.0]:+.0%}) and stock-outs fell from {rates.loc['Mar-Apr (before)', 1.0]:.1%} to "
               f"{rates.loc['Jun-Aug (after reset)', 1.0]:.1%} - but they still stock out {rates.loc['Jun-Aug (after reset)', 1.0] / rates.loc['Jun-Aug (after reset)', 0.0]:.1f}x "
               f"as often as the rest ({rates.loc['Jun-Aug (after reset)', 0.0]:.1%}). The fix under-corrected because it was based on sales. "
               f"(Our demand estimate is conservative, so the true shortfall in cover is larger.) "
               f"Action: plan from estimated demand (demand_est), never from raw sales.")
    return dict(name="08_censored_demand_reset", question=q, fig=fig, insight=insight,
                table=pd.concat([rates.T.add_prefix("stockout_rate "), cover, change.rename("median_level_change")], axis=1))


def eda_lost_sales(m):
    q = "Where is revenue being lost to empty shelves?"
    by_cat = m.groupby("category").lost_sales_est.sum().sort_values()
    by_store = m.groupby("store_id").lost_sales_est.sum().sort_values()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, s, color_map, title in [(a1, by_cat, CAT_COLOR, "By category"), (a2, by_store, STORE_COLOR, "By store")]:
        ax.barh(s.index, s / 1e5, color=[color_map[k] for k in s.index], height=0.6)
        for y, v in enumerate(s):
            ax.text(v / 1e5, y, f"  {rupees(v)}", va="center", color=INK2, fontsize=9)
        ax.set_xlabel("Estimated lost sales (Rs lakh)"); ax.set_title(title)
        ax.grid(axis="y", visible=False); ax.grid(axis="x")
        ax.set_xlim(0, s.max() / 1e5 * 1.35)
    suptitle(fig, q); fig.tight_layout()

    total = m.lost_sales_est.sum()
    insight = (f"Estimated lost sales: {rupees(total)} over 6 months ({total / m.revenue.sum():.1%} of revenue) - a conservative figure, "
               f"because demand beyond an empty shelf is only partly visible. {by_cat.index[-1]} alone accounts for "
               f"{by_cat.iloc[-1] / total:.0%}; {by_store.index[-1]} is the store losing most ({rupees(by_store.iloc[-1])}). "
               f"Action: prioritise the action list by rupees at risk, not by alphabetical order or probability alone.")
    return dict(name="09_lost_sales", question=q, fig=fig, insight=insight,
                table=pd.DataFrame({"lost_sales_by_category": by_cat.sort_values(ascending=False)}))


def eda_temperature(m):
    q = "Does heat drive beverage demand?"
    bev = m[m.category == "Beverages"].groupby(["store_id", "date"]).agg(demand=("demand_est", "sum"), temp=("temp_c", "first")).reset_index()
    s = bev[bev.store_id == "S02"]
    rho = stats.spearmanr(s.temp, s.demand)[0]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(s.temp, s.demand, s=16, color=SLOTS[0], alpha=0.6, edgecolor="none")
    fit = np.polyfit(s.temp, s.demand, 1)
    xs = np.linspace(s.temp.min(), s.temp.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=INK2, lw=1.5, ls="--")
    ax.set_xlabel("Daily temperature (C)"); ax.set_ylabel("Beverage demand (units/day)")
    ax.set_title(f"S02 Chennai Hypermarket - Spearman rho = {rho:.2f}")
    ax.grid(axis="x")
    suptitle(fig, q); fig.tight_layout()
    bev_so = m[m.category == "Beverages"].groupby(m.date.dt.month_name()).stockout_flag.mean()
    hot = m[m.city == "Chennai"].groupby(m.date.dt.month_name()).temp_c.mean()
    insight = (f"Yes - in S02, each extra degree adds about {fit[0]:.0f} beverage units a day (rho = {rho:.2f}). "
               f"Beverage stock-outs peak in {bev_so.idxmax()} ({bev_so.max():.0%} of days); Chennai is hottest in {hot.idxmax()} ({hot.max():.1f} C). "
               f"Action: include the weather forecast in beverage orders (see test T4 for all stores).")
    return dict(name="10_temperature_beverages", question=q, fig=fig, insight=insight, table=None)


# ================================================================ statistics
def rank_biserial(diff):
    d = diff[diff != 0]
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def mantel_haenszel(df, exposure, outcome, stratum):
    """Common odds ratio and CMH chi-square across strata (removes confounding by the stratum)."""
    num = den = dev = var = 0.0
    for _, g in df.groupby(stratum):
        a = ((g[exposure] == 1) & (g[outcome] == 1)).sum(); b = ((g[exposure] == 1) & (g[outcome] == 0)).sum()
        c = ((g[exposure] == 0) & (g[outcome] == 1)).sum(); d = ((g[exposure] == 0) & (g[outcome] == 0)).sum()
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n; den += b * c / n
        dev += a - (a + b) * (a + c) / n
        var += (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1))
    chi2 = (abs(dev) - 0.5) ** 2 / var
    return num / den, chi2, stats.chi2.sf(chi2, 1)


def test_promotions(m):
    pairs = promo_pairs(m)
    w = stats.wilcoxon(pairs.promo, pairs.normal, alternative="greater")
    r = rank_biserial((pairs.promo - pairs.normal).values)
    per_cat = []
    for cat, g in pairs.groupby("category"):
        t = stats.wilcoxon(g.promo, g.normal, alternative="greater")
        per_cat.append(dict(category=cat, pairs=len(g), median_lift=g.lift.median(),
                            p_value=t.pvalue, p_bonferroni=min(1, t.pvalue * pairs.category.nunique())))
    per_cat = pd.DataFrame(per_cat).sort_values("median_lift", ascending=False)
    sig = per_cat[per_cat.p_bonferroni < ALPHA].category.tolist()
    row = dict(
        id="T1", question="Do promotions significantly increase sales?",
        H0="For the same store x product, median daily units on promo days = on non-promo days",
        H1="Median daily units are higher on promo days (one-sided)",
        test="Wilcoxon signed-rank, paired by store x product (+ per category, Bonferroni)",
        why="Pairing compares a product with itself, so the mix of which products get promoted cannot fake a lift; "
            "daily unit counts are skewed, so a rank test is safer than a t-test.",
        n=f"{len(pairs)} store-product pairs", statistic=f"W = {w.statistic:,.0f}", p_value=w.pvalue,
        effect=f"median lift {pairs.lift.median():+.0%}; rank-biserial r = {r:.2f}; {(pairs.lift > 0).mean():.0%} of pairs gain",
        conclusion="Reject H0" if w.pvalue < ALPHA else "Fail to reject H0",
        interpretation=f"Promotions raise units sold, and the effect is large (median {pairs.lift.median():+.0%}). "
                       f"Significant in {len(sig)}/{len(per_cat)} categories after Bonferroni: {', '.join(sig)}. "
                       f"Strongest: {per_cat.category.iloc[0]} ({per_cat.median_lift.iloc[0]:+.0%}).",
        action="Every planned promotion must increase that week's order by the category lift; promotion days are a required forecasting feature.")
    return row, per_cat


def test_store_types(m):
    g = m[m.sparse_history_flag == 0]
    common = g.groupby("product_id").store_type.nunique()
    common = common[common == 3].index
    combo = g[g.product_id.isin(common)].groupby(["store_type", "store_id", "product_id"]).demand_est.mean().reset_index()
    groups = [combo[combo.store_type == f].demand_est for f in FORMATS]
    h = stats.kruskal(*groups)
    n, k = len(combo), len(FORMATS)
    eps2 = (h.statistic - k + 1) / (n - k)
    post = []
    for a, b in combinations(FORMATS, 2):
        u = stats.mannwhitneyu(combo[combo.store_type == a].demand_est, combo[combo.store_type == b].demand_est)
        post.append(dict(pair=f"{a} vs {b}", U=u.statistic, p_value=u.pvalue, p_bonferroni=min(1, u.pvalue * 3)))
    post = pd.DataFrame(post)
    # Robustness: same products as blocks (Friedman), and ANOVA on log demand.
    blocks = combo.groupby(["product_id", "store_type"]).demand_est.mean().unstack()
    fr = stats.friedmanchisquare(*[blocks[f] for f in FORMATS])
    an = stats.f_oneway(*[np.log1p(x) for x in groups])
    med = combo.groupby("store_type").demand_est.median()
    row = dict(
        id="T2", question="Does mean demand differ across store types?",
        H0="Daily demand per product has the same distribution in Express, Supermarket and Hypermarket stores",
        H1="At least one store type differs",
        test="Kruskal-Wallis + pairwise Mann-Whitney (Bonferroni); robustness: Friedman (product as block), ANOVA on log demand",
        why=f"Three groups and skewed demand -> rank test. Restricted to the {len(common)} products sold in ALL formats, "
            f"so differences in assortment cannot create a false difference.",
        n=f"{n} store-product combinations ({len(common)} products)", statistic=f"H = {h.statistic:.1f}", p_value=h.pvalue,
        effect=f"epsilon^2 = {eps2:.2f}; median units/day: " + ", ".join(f"{f} {med[f]:.1f}" for f in FORMATS),
        conclusion="Reject H0" if h.pvalue < ALPHA else "Fail to reject H0",
        interpretation=f"Store format matters a lot: a Hypermarket sells {med['Hypermarket'] / med['Express']:.1f}x an Express store's volume of the same product. "
                       f"Pairwise (Bonferroni): " + "; ".join(f"{r.pair} p={fmt_p(r.p_bonferroni)}" for r in post.itertuples()) +
                       f". Friedman p = {fmt_p(fr.pvalue)}, ANOVA(log) p = {fmt_p(an.pvalue)} - same conclusion.",
        action="Reorder levels and safety stock must be set per store (format), never copied across stores; store_type is a required model feature.")
    return row, post


def test_stockout_promo(m):
    tab = pd.crosstab(m.promo_flag, m.stockout_flag)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    a, b, c, d = tab.loc[1, 1], tab.loc[1, 0], tab.loc[0, 1], tab.loc[0, 0]
    odds = (a * d) / (b * c)
    v = np.sqrt(chi2 / tab.values.sum())
    or_mh, chi_mh, p_mh = mantel_haenszel(m, "promo_flag", "stockout_flag", ["store_id", "category"])
    rates = m.groupby("promo_flag").stockout_flag.mean()
    row = dict(
        id="T3", question="Is stock-out frequency associated with promotion status?",
        H0="Stock-out status is independent of promotion status",
        H1="Stock-out status depends on promotion status",
        test="Chi-square test of independence (2x2) + odds ratio, Cramer's V; Mantel-Haenszel by store x category",
        why="Both variables are binary. With ~70k rows any tiny difference is 'significant', so the odds ratio is what we interpret. "
            "Mantel-Haenszel checks that it is not just because promoted products are in stock-out-prone stores/categories.",
        n=f"{tab.values.sum():,} store-product-days", statistic=f"chi2 = {chi2:,.1f} (df = {dof})", p_value=p,
        effect=f"stock-out rate {rates[1]:.1%} (promo) vs {rates[0]:.1%}; OR = {odds:.2f}; Cramer's V = {v:.3f}; MH-adjusted OR = {or_mh:.2f}",
        conclusion="Reject H0" if p < ALPHA else "Fail to reject H0",
        interpretation=f"Promotion days have {odds:.1f}x the odds of a stock-out ({or_mh:.1f}x after adjusting for store and category, MH p = {fmt_p(p_mh)}). "
                       f"Cramer's V is small because stock-outs are rare overall, but the business effect is large: stores are not ordering for promotions.",
        action="Link the promotions calendar to replenishment: raise stock before a promotion starts; promo_days_next_7 is a key stock-out feature.")
    return row, tab


def test_temperature(m):
    bev = m[m.category == "Beverages"].groupby(["store_id", "date"]).agg(demand=("demand_est", "sum"), temp=("temp_c", "first")).reset_index()
    per_store = []
    for s, g in bev.groupby("store_id"):
        r = stats.spearmanr(g.temp, g.demand)
        per_store.append(dict(store_id=s, rho=r.statistic, p_value=r.pvalue, p_bonferroni=min(1, r.pvalue * 6)))
    per_store = pd.DataFrame(per_store)
    best = per_store.loc[per_store.rho.idxmax()]
    sig = (per_store.p_bonferroni < ALPHA).sum()
    row = dict(
        id="T4", question="Does temperature drive beverage demand?",
        H0="No monotonic association between daily temperature and beverage demand (rho = 0)",
        H1="rho != 0",
        test="Spearman rank correlation, within each store (Bonferroni over 6 stores)",
        why="The relation need not be linear; testing within a store removes store-size differences.",
        n=f"{len(bev)} store-days", statistic=f"median rho = {per_store.rho.median():.2f}", p_value=per_store.p_value.max(),
        effect=f"rho ranges {per_store.rho.min():.2f} to {per_store.rho.max():.2f}; significant in {sig}/6 stores",
        conclusion="Reject H0" if sig == 6 else f"Reject H0 in {sig}/6 stores",
        interpretation=f"Hotter days sell more beverages in every store (strongest {best.store_id}, rho = {best.rho:.2f}). "
                       "p-value shown is the largest (weakest) of the six stores. Temperature also moves with season, so part of this is the summer effect.",
        action="Use the temperature forecast for beverage and frozen orders in summer.")
    return row, per_store


def test_supplier(po):
    rec = po[po.status == "RECEIVED"]
    tab = pd.crosstab(rec.supplier_id, rec.is_late)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    worst = rec.groupby("supplier_id").is_late.mean().idxmax()
    t2 = pd.crosstab(rec.supplier_id == worst, rec.is_late)
    odds = (t2.loc[True, 1] * t2.loc[False, 0]) / (t2.loc[True, 0] * t2.loc[False, 1])
    rates = rec.groupby(rec.supplier_id == worst).is_late.mean()
    row = dict(
        id="T5", question="Is late delivery associated with the supplier?",
        H0="Late delivery rate is the same for all 10 suppliers",
        H1="Late delivery rate differs by supplier",
        test="Chi-square test of independence (10 x 2) + odds ratio for the worst supplier vs the rest",
        why="Categorical supplier vs binary late/on-time outcome.",
        n=f"{len(rec):,} received purchase orders", statistic=f"chi2 = {chi2:,.1f} (df = {dof})", p_value=p,
        effect=f"{worst} late {rates[True]:.0%} vs {rates[False]:.0%} others; OR = {odds:.1f}; Cramer's V = {v:.2f}",
        conclusion="Reject H0" if p < ALPHA else "Fail to reject H0",
        interpretation=f"{worst} is the outlier: {odds:.1f}x the odds of a late delivery. This explains why its category stocks out most.",
        action=f"Use {worst}'s effective lead time (nominal + average delay) in reorder calculations and raise it with the supplier.")
    return row, tab


def test_censoring(m):
    g = censoring_groups(m)
    after = g[g.date >= "2026-06-01"].groupby(["store_id", "product_id", "censored_in_may"]).stockout_flag.mean().reset_index()
    x, y = after[after.censored_in_may == 1].stockout_flag, after[after.censored_in_may == 0].stockout_flag
    u = stats.mannwhitneyu(x, y, alternative="greater")
    r = 2 * u.statistic / (len(x) * len(y)) - 1
    before = g[g.date < "2026-05-01"].groupby("censored_in_may").stockout_flag.mean()
    row = dict(
        id="T6", question="Do products that ran out in May keep running out after the June reorder recalibration?",
        H0="Jun-Aug stock-out rate is the same for combinations that stocked out in May and those that did not",
        H1="It is higher for combinations that stocked out in May (one-sided)",
        test="Mann-Whitney U on combination-level Jun-Aug stock-out rates",
        why="One value per store x product avoids treating 90 correlated days as independent; rates are skewed and many are 0.",
        n=f"{len(x)} vs {len(y)} combinations", statistic=f"U = {u.statistic:,.0f}", p_value=u.pvalue,
        effect=f"mean Jun-Aug stock-out rate {x.mean():.1%} vs {y.mean():.1%} (Mar-Apr: {before[1.0]:.1%} vs {before[0.0]:.1%}); rank-biserial r = {r:.2f}",
        conclusion="Reject H0" if u.pvalue < ALPHA else "Fail to reject H0",
        interpretation="Stock-outs are chronic: the combinations that ran out in May kept running out after the 1 June reset. "
                       "The reset set levels from May sales, which were capped by empty shelves, so it under-corrected "
                       "(see EDA chart 8 for the cover in days of sales vs days of estimated demand).",
        action="Recalibrate reorder levels on estimated demand (demand_est), and review every combination that stocked out in May.")
    return row, after


def test_weekend(m):
    g = m[m.sparse_history_flag == 0]
    per = g.groupby(["store_type", "store_id", "product_id", "weekend"]).units_sold.mean().unstack()
    per = per.dropna()
    per = per[per[0] > 0]
    res = []
    for f, x in per.groupby("store_type"):
        w = stats.wilcoxon(x[1], x[0])
        res.append(dict(store_type=f, pairs=len(x), median_uplift=(x[1] / x[0] - 1).median(),
                        p_value=w.pvalue, p_bonferroni=min(1, w.pvalue * 3)))
    res = pd.DataFrame(res)
    w_all = stats.wilcoxon(per[1], per[0])
    row = dict(
        id="T7", question="Is weekend demand different from weekday demand?",
        H0="For the same store x product, median daily units on weekends = on weekdays",
        H1="They differ (two-sided)",
        test="Wilcoxon signed-rank, paired by store x product, overall and per store format (Bonferroni)",
        why="Paired within product, skewed counts.",
        n=f"{len(per)} store-product pairs", statistic=f"W = {w_all.statistic:,.0f}", p_value=w_all.pvalue,
        effect="median weekend uplift: " + ", ".join(f"{r.store_type} {r.median_uplift:+.0%}" for r in res.itertuples()),
        conclusion="Reject H0" if w_all.pvalue < ALPHA else "Fail to reject H0",
        interpretation="The weekend effect is real but its size depends on the format: " +
                       "; ".join(f"{r.store_type} p={fmt_p(r.p_bonferroni)}" for r in res.itertuples()) + ".",
        action="Deliveries should arrive before the weekend peak in the formats that have one; weekend days in the next 7 is a forecasting feature.")
    return row, res


# =================================================================== reports
def write_eda_report(results, kpi_table, kpi_notes, monthly):
    lines = ["# EDA - Business Insights", "",
             "_Generated by `src/eda_stats.py` from `data/processed/master_dataset.csv`. Every number below is computed, not typed._", "",
             "## Headline KPIs (1 Mar - 31 Aug 2026)", "",
             "| KPI | All stores | Definition |", "|---|---|---|"]
    for k, v in kpi_table.loc["All stores"].items():
        if "Rs" in k:
            val = rupees(v)
        elif "Rate" in k or "Lift" in k or "%" in k:
            val = f"{v:.1%}"
        elif "Units" in k:
            val = f"{v:,.0f}"
        else:
            val = f"{v:.1f}"
        lines.append(f"| {k} | {val} | {kpi_notes[k]} |")
    lines += ["", "Per-store KPIs: `reports/kpi_summary.csv`.", "",
              "### Monthly trend", "", "| Month | Revenue | MoM growth | Units | Stock-out rate | Est. lost sales |", "|---|---|---|---|---|---|"]
    for p, r in monthly.iterrows():
        g = "-" if pd.isna(r.revenue_growth_mom) else f"{r.revenue_growth_mom:+.1%}"
        lines.append(f"| {p} | {rupees(r.revenue)} | {g} | {r.units:,.0f} | {r.stockout_rate:.1%} | {rupees(r.lost_sales)} |")
    lines += ["", "## Insights (one business question per chart)", ""]
    for i, r in enumerate(results, 1):
        lines += [f"### {i}. {r['question']}", "", f"![{r['question']}](figures/{r['name']}.png)", "", r["insight"], ""]
    (REPORTS / "eda_insights.md").write_text("\n".join(lines), encoding="utf-8")


def write_stats_report(rows, details):
    lines = ["# Statistical Tests", "",
             "_Generated by `src/eda_stats.py`. Significance level alpha = 0.05. Where several tests answer one question, "
             "p-values are Bonferroni-adjusted. Effect sizes are reported because with ~70k rows almost any difference is 'significant' - "
             "the effect size says whether it matters to the business._", "",
             "## Summary", "", "| # | Business question | Test | p-value | Effect | Decision |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['question']} | {r['test'].split(',')[0].split('+')[0].strip()} | {fmt_p(r['p_value'])} | {r['effect']} | {r['conclusion']} |")
    for r in rows:
        lines += ["", f"## {r['id']}. {r['question']}", "",
                  f"- **H0:** {r['H0']}", f"- **H1:** {r['H1']}",
                  f"- **Test:** {r['test']}", f"- **Why this test:** {r['why']}",
                  f"- **Sample:** {r['n']}", f"- **Statistic:** {r['statistic']}",
                  f"- **p-value:** {fmt_p(r['p_value'])}", f"- **Effect size:** {r['effect']}",
                  f"- **Decision:** {r['conclusion']}",
                  f"- **Business interpretation:** {r['interpretation']}",
                  f"- **What the manager should do:** {r['action']}"]
        d = details.get(r["id"])
        if d is not None:
            lines += ["", md_table(d)]
    (REPORTS / "statistical_tests.md").write_text("\n".join(lines), encoding="utf-8")


def run_eda(m, po):
    return [eda_category_revenue(m), eda_store_trend(m), eda_promotions(m), eda_weekend(m),
            eda_volatility(m), eda_stockout_heatmap(m), eda_supplier(m, po), eda_censoring(m),
            eda_lost_sales(m), eda_temperature(m)]


def run_tests(m, po):
    out = [test_promotions(m), test_store_types(m), test_stockout_promo(m), test_temperature(m),
           test_supplier(po), test_censoring(m), test_weekend(m)]
    rows = [r for r, _ in out]
    details = {r["id"]: d for r, d in out if r["id"] in ("T1", "T2", "T4", "T7")}
    return rows, details


def main():
    matplotlib.use("Agg")   # script mode: write files, no windows (the notebook keeps its inline backend)
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = load_master()
    po = pd.read_csv(PROCESSED / "purchase_orders_clean.csv", parse_dates=["order_date", "expected_date", "received_date"])

    print("KPIs ...")
    kpi_table, kpi_notes = kpis(m)
    kpi_table.to_csv(REPORTS / "kpi_summary.csv")
    monthly = monthly_trend(m)

    print("EDA charts ...")
    results = run_eda(m, po)
    for r in results:
        r["fig"].savefig(FIGURES / f"{r['name']}.png")
        plt.close(r["fig"])
        print(f"  {r['name']}: {r['insight'][:110]}...")
    volatility_table(m).to_csv(REPORTS / "volatility_by_product.csv", index=False)
    write_eda_report(results, kpi_table, kpi_notes, monthly)

    print("Statistical tests ...")
    rows, details = run_tests(m, po)
    pd.DataFrame(rows).to_csv(REPORTS / "statistical_tests.csv", index=False)
    for r in rows:
        print(f"  {r['id']} p={fmt_p(r['p_value'])} | {r['effect']}")
    write_stats_report(rows, details)


if __name__ == "__main__":
    main()
