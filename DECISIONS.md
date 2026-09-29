# StockSense — Decision Playbook

Each decision is written as **Decision → Why → What goes wrong if you don't**. Judges ask "why?" far more than "what?", and every team member must be able to defend all of these.

---

## 0. Framing the problem (do this before touching data)

### D0.1 The prediction moment: end of day *t*, per store × product
Every evening after closing, for every store × product, the system predicts demand for days *t+1 … t+7* and whether stock will hit zero in that window.
**Why:** this is when a manager can still raise an order. Fixing the moment also fixes what is "known", which is the entire basis of the leakage rule.
**If skipped:** teams end up with features that silently contain the future and a model that looks brilliant but can't be deployed.

### D0.2 Both models share one origin row
One row = (store, product, origin date *t*). Model 1 target = sum of demand over *t+1..t+7*. Model 2 target = 1 if closing stock reaches 0 on any day in *t+1..t+7*.
**Why:** the two outputs feed the same recommendation card, so they must describe the same week.

### D0.3 The metric that matters is rupees, not accuracy
Rank actions by **expected loss = stock-out probability × expected shortfall units × unit margin**, not by probability alone.
**Why:** a 90% risk on a ₹15 buttermilk matters less than a 60% risk on 5 kg rice at a hypermarket. This is what "business solution, not just a model" means in practice.

---

## 1. Round 1 — Cleaning, integration, EDA, statistics

### D1.1 Build the master table from inventory, not from transactions
Create the full date × store × product grid (from each product's launch date), left-join aggregated transactions onto it, and fill missing sales with 0.
**Why:** aggregated transactions only contain days with a sale. Zero-sale days vanish, which inflates average demand and makes it impossible to see stock-out days. Inventory has a row every day.
**If skipped:** rolling means are biased upward and stock-outs are invisible.

### D1.2 Parse dates with explicit formats, never infer
Roughly 1.2% of transaction dates are `DD-MM-YYYY`. Parse `%Y-%m-%d` first, then `%d-%m-%Y` for the failures.
**Why:** automatic inference can read 03-08-2026 as March 8 instead of August 3 — silently moving sales into the wrong month.

### D1.3 Standardise IDs and text before any join, and check row counts after every merge
Upper-case and strip `store_id`/`product_id`; map categories (`beverage`, `BEVERAGES`, `Beverages␠` → `Beverages`; `Grocery` → `Groceries`); map city alias `Trichy` → `Tiruchirappalli`; title-case `store_type`.
**Why:** a failed join doesn't raise an error — it produces NaNs. The city alias alone wipes out weather for an entire store.
**Habit:** assert `len(after) == len(before)` and assert zero unmatched keys after every merge. Put that assertion in the Data Quality Report — judges love it.

### D1.4 Duplicates: remove exact duplicates only
Drop rows where the entire record repeats (same `transaction_id` and same content). Keep one copy of duplicated inventory/product/weather rows.
**Why:** two different lines can legitimately share a product, store and hour. Only identical records are true duplicates.

### D1.5 Bad quantities: verify against inventory before deciding
- `quantity ≤ 0` → returns/voids, excluded from demand, counted in the report.
- Extreme quantities (hundreds of units in one line) → compare the day's transaction total with `inventory.sold_qty`. If inventory doesn't agree, it's a keying error → drop.

**Why:** cross-system reconciliation is evidence; an IQR rule is a guess. A blanket outlier cap would also clip genuine volume (milk at the Chennai hypermarket legitimately sells ~190 units/day).

### D1.6 Prices: enforce `0 < selling_price ≤ MRP`
Missing price → median for that store × product × date (or × promo status). Prices ≈10× MRP → replace the same way.
**Why:** revenue, discount features and lost-sales value all depend on price.

### D1.7 Inventory arithmetic: trust the physical count
Recompute `expected_closing = opening + received − sold`. Where it disagrees, use the next day's `opening_stock` (the physical count) to decide which figure is right. Negative closing stock is impossible → replace with the recomputed value. Missing inventory rows → rebuild from the previous day's closing plus transactions and purchase-order receipts.
**Why:** stock-out labels come from `closing_stock`, so errors here become label noise in Model 2.

### D1.8 Weather: invalidate, then interpolate within city
Temperatures outside 10–45 °C are sensor errors → set to missing. Fill missing temperature and rain by time interpolation within the city.
**Why:** weather is smooth over time within a city; a global mean ignores the Chennai/Coimbatore climate gap.

### D1.9 Leave blank `customer_id` blank
**Why:** it means "not a loyalty member". It isn't used for demand, so imputing it would add fiction without value. Say so in the report — it shows judgement, not negligence.

### D1.10 Operational definition of stock-out
`stockout_flag = 1` if reconciled `closing_stock == 0` at end of day.
**Why:** it is objective, directly observable, and matches what a manager sees (an empty shelf at close). Mention the alternative (closing stock < one day of demand = "near stock-out") and why you didn't use it as the main label: it depends on a demand estimate, which makes the label partly model-generated.

### D1.11 Correct for censored demand (the most important insight in the dataset)
On stock-out days, **sales < demand** — customers wanted more but the shelf was empty. Create `demand_est`:
- normal day → `demand_est = sold`
- stock-out day → `demand_est = max(sold, typical demand)`, where typical demand = mean of the last 4 same-weekday non-stock-out days (or a 7-day non-stock-out rolling mean)

Use `demand_est` for the forecasting target and demand features, and `(demand_est − sold) × price` for **Estimated Lost Sales**.
**Why:** training on raw sales teaches the model that stores which run out have "low demand", so it forecasts low, the store orders less, and it runs out again. The dataset contains exactly this spiral: on 1 June the planners recalibrated reorder levels using the previous 30 days of *sales*, which locked under-stocked stores into low levels.
**Proof:** the answer key contains true demand — show the gap between raw sales and `demand_est` versus truth.

### D1.12 Sparse-history products: keep and flag
Add `history_days`. Keep them in the table (their lag features will be NaN, which XGBoost handles).
**Why:** dropping them means the new launches get no recommendation at all — a store manager would notice.

### D1.13 EDA: one business question per chart
| Question | Analysis | What to look for |
|---|---|---|
| Where is revenue? | Category share + Pareto of products | ~20% of products → majority of revenue |
| Growing/declining stores? | Weekly units by store, indexed to week 1 | Clear growers and decliners |
| Do promotions work? | Within-product promo vs non-promo daily units | Lift differs strongly by category |
| Weekend behaviour? | Weekday vs weekend by store format | Format matters, not just the day |
| Volatile products? | CV of daily demand | High CV → higher safety stock |
| Who keeps stocking out? | Heatmap store × category of stock-out rate | Chronic store/category pairs |
| Supplier reliability? | Late-delivery rate by supplier (purchase orders) | An unreliable supplier explains a category |

### D1.14 Statistical tests: pick tests that match the data, and report effect sizes
| Question | Test | Why this test |
|---|---|---|
| Do promotions increase sales? | **Wilcoxon signed-rank**, paired per store × product (mean units on promo days vs non-promo days) | Pairing removes product mix (a pooled t-test is confounded by *which* products get promoted); daily counts are skewed, so non-parametric |
| Does demand differ across store types? | **Kruskal–Wallis** on daily demand of products carried by all formats, then pairwise Mann–Whitney with Bonferroni | Three groups, skewed data; restricting to common products stops assortment from faking a difference |
| Is stock-out associated with promotion? | **Chi-square** on a 2×2 table + odds ratio / Cramér's V | Both variables are binary; with ~70k rows the p-value is always tiny, so the effect size is what you interpret |
| (bonus) Temperature vs beverage demand | **Spearman**, within a single store | Monotonic, not necessarily linear; within-store removes store-size confounding |

Always write: business question → H0 → H1 → test → p-value → effect size → **what the manager should do differently**.

---

## 2. Round 2 — Features and models

### D2.1 Feature legality rule
A feature is legal if it would be known at end of day *t*:
| Allowed | Why |
|---|---|
| Lags and rolling stats of `demand_est` up to day *t* | Past |
| Stock on hand at close of *t*; open POs due within 7 days | Known to the store |
| Calendar for *t+1..t+7*: weekend days count, holiday/festival in window, days to next festival, payday days (1st–5th) in window | Calendars are published in advance |
| Planned promotions in *t+1..t+7* (from `promotions_calendar.csv`) | Marketing plans them in advance |
| Weather for *t+1..t+7* | **Only as a forecast.** Safest: use recent observed temperature (last 3 days). If you use the future rows, state that they stand in for a weather forecast |

**Not allowed:** anything computed over a window that includes *t+1* or later, and `promotion_flag` taken from future transactions (use the calendar instead).
**Implementation trap:** always compute rolling features within each store × product group, and check your shift. A `rolling(7).mean()` without grouping leaks across products.

### D2.2 Feature set
| Group | Features | Why it matters |
|---|---|---|
| Recent demand | lag_1, lag_7, lag_14, rolling_mean_7/14/28, rolling_std_7, trend = rm7 / rm28 | Level, weekly pattern, momentum |
| Calendar (future window) | weekend_days_next7, festival_next7, days_to_festival, holiday_next7, payday_days_next7 | Festivals drive pre-festival buying, especially pooja items, ghee, snacks |
| Promotion (future window) | promo_days_next7, max_discount_next7, days_since_last_promo | Promotions lift demand, and a post-promotion dip follows for long shelf-life items |
| Price | price_ratio (current / 28-day median), price_change_30d | Several price hikes occur mid-period |
| Inventory | closing_stock, incoming_next7, days_of_cover = (stock + incoming) / rm7, reorder_gap = stock − reorder_level | Model 2's strongest signals |
| History of failure | stockout_days_last14, stockout_rate_last28 | Chronic problems repeat |
| Supplier | lead_time, supplier_late_rate (computed on the training period only) | Unreliable suppliers → effective lead time is longer |
| Static | store_type, category, shelf_life, avg_daily_customers | Lets one global model share learning |
| Weather | temp_last3, rain_last3 | Beverages/ice cream respond to heat |

### D2.3 One global model, not 387 small models
**Why:** each store × product has only ~150 training days. A global model learns "promotions lift snacks" once and applies it everywhere, and it can still predict new products from their category/store features.

### D2.4 Time-based split with no overlapping target windows
| Split | Origin dates | Target window ends |
|---|---|---|
| Warm-up (features only) | Mar 1 – Mar 28 | — |
| Train | Mar 29 – Jul 12 | Jul 19 |
| Validation | Jul 20 – Aug 3 | Aug 10 |
| Test | Aug 11 – Aug 24 | Aug 31 |
| Final forecast | Aug 31 → predicts Sep 1–7 | scored with the answer key |

**Why the gaps:** a training row with origin Jul 17 has a target covering Jul 18–24. If validation started on Jul 18, validation days would sit inside training targets — leakage through the label. Ending training origins 7 days before validation starts removes it.
Hyperparameters are tuned on validation; the test set is used **once**. Then refit on everything and forecast Sep 1–7.

### D2.5 Model 1 (demand) — algorithm choice
| Model | Role | Verdict |
|---|---|---|
| 7-day mean × 7 | Mandatory baseline | Must be beaten |
| Linear Regression (on log1p target) | Interpretable benchmark | Fast, weak on interactions (promo × category) |
| Random Forest | Strong benchmark | Good, but can't extrapolate growth |
| **XGBoost** | **Selected** | Best with mixed features, handles NaN for new products, interactions, fast |
| KNN / SVM | Not used | KNN degrades on ~60k rows with mixed scales; kernel SVR training is roughly quadratic in rows. State this rather than skipping silently |
| Naive Bayes | N/A | It's a classifier |

Use `log1p(target)` (or XGBoost's `reg:tweedie` objective) because demand is a skewed count spanning ~1 to ~2,000 units/week.

### D2.6 Model 1 metrics — WAPE/MAE first, not MAPE
Report MAE, RMSE, MAPE, R² as required, but **select on WAPE = Σ|error| / Σactual** (plus MAE).
**Why:** MAPE explodes on slow movers (actual 2, forecast 4 = 100% error) and would push you to optimise the least important products. WAPE weights errors by volume, which tracks money. Also report **bias** (mean of forecast − actual): systematic under-forecasting causes stock-outs even when MAE looks fine.

### D2.7 Model 2 (stock-out risk)
- **Target:** any stock-out in *t+1..t+7* (≈29% positive in this data).
- **Stacking:** feed Model 1's forecast in as a feature → `coverage = (stock + incoming) / forecast_7d`. Use **out-of-time** forecasts (predictions made by a model that never saw those rows); otherwise Model 2 trusts an unrealistically perfect forecast.
- **Algorithms:** Naive Bayes (baseline) vs Decision Tree vs Random Forest vs **XGBoost (selected)**. Logistic regression is not on the permitted list — don't use it.
- **Imbalance:** `scale_pos_weight` / `class_weight='balanced'`. Avoid SMOTE — synthetic interpolated rows across time add noise and aren't needed at 29% positives.
- **Calibration (critical):** the business rules are applied to *probabilities* (≥0.70 High, 0.40–0.70 Medium). Class reweighting inflates probabilities, so calibrate on the validation set (isotonic or sigmoid) and show a reliability curve + Brier score.
- **Metric priority:** Recall on the positive class and PR-AUC first (a missed stock-out costs revenue and trust; a false alarm costs a phone call). Report accuracy, precision, F1, ROC-AUC and the confusion matrix at both 0.40 and 0.70 as required. Explain why accuracy is misleading here: predicting "no stock-out" for everything gives ~71%.

### D2.8 Explainability
- **Global:** permutation importance on the **test** set (it measures what the model really relies on for unseen data; built-in gain importance favours high-cardinality features).
- **Local (per recommendation card):** XGBoost's per-prediction contributions (`pred_contribs=True`). Group features into plain-language drivers ("Promotion planned — 20% off", "Festival in 3 days", "Stock covers only 1.8 days", "Supplier late 36% of the time") and show each driver's share of the positive push, like the brief's +31% example.
- **Fallback if judges insist on permutation-only:** for a single row, reset one feature group to its "normal" value (e.g. no promotion, 28-day average) and report how much the probability drops.

---

## 3. Round 3 — Action layer and prototype

### D3.1 Safety stock from forecast error, not a fixed %
`safety_stock = z × σ₇ × √((lead_time + review_days) / 7)`
σ₇ = standard deviation of 7-day forecast errors on validation, per category (or per product with enough history); z = 1.65 (95% service) normally, 1.28 for perishables.
**Why:** volatile products and long/unreliable lead times need more buffer. A fixed 20% over-stocks stable items and under-stocks volatile ones.

### D3.2 Effective lead time includes supplier reliability
`effective_lead = lead_time + average lateness of that supplier` (from purchase orders).
**Why:** one supplier is late on about a third of its deliveries. Ignoring that is why an entire category keeps stocking out.

### D3.3 Reorder quantity
```
recommended_stock = forecast_7d + safety_stock
reorder_qty = max(0, recommended_stock − current_stock − incoming_within_7d)
→ rounded up to case pack
```
Count only open POs expected to arrive **within** the window.

### D3.4 Perishable cap
For shelf life < 7 days (milk 3, curd 5, bread 4, batter 4), cover only `min(7, shelf_life − 1)` days of forecast.
**Why:** the brief's formula would order a week of 3-day milk — you'd convert a stock-out problem into a waste problem. Mention this in the pitch; it shows business sense.

### D3.5 Decision matrix (risk × reorder)
| Risk | Reorder > 0 | Reorder = 0 |
|---|---|---|
| HIGH | **Raise order today** | **Expedite / confirm open PO** (stock is coming but may arrive late) |
| MEDIUM | Add to next order cycle | Monitor daily |
| LOW | Order at normal cycle | If days of cover > 2 × (lead + 7) → **Overstock: pause orders / markdown** |

**Why:** the classifier and the reorder formula can disagree; the matrix turns every combination into an unambiguous instruction, and the overstock row addresses the brief's second business problem (excess inventory).

### D3.6 Prioritise by money
Sort the Manager Action Centre by expected lost revenue (or margin) at risk, not alphabetically or by probability.

### D3.7 Prototype: Streamlit
Six pages matching the brief's minimum sections. What-if sliders: discount % (re-predict with changed promo features), supplier delay (+days to lead → recompute incoming and reorder), festival demand multiplier.
**Why Streamlit:** pure Python, same code as the notebooks, one person can build it in hours, and the pipeline stays reproducible.

### D3.8 Prove it on unseen data
Forecast Sep 1–7 from origin Aug 31 and score it against `answer_key_DO_NOT_COMMIT/true_demand_and_holdout.csv`. Show: forecast WAPE for that week, how many of the actual stock-outs your HIGH list caught, and the lost revenue avoided if the recommended orders had been placed (compare with `lost_units_if_no_action`). That single slide answers "why should we trust this?"

---

## 4. Pitch storyline (5 minutes)
1. **Problem in rupees:** current rule-based ordering loses X% of demand; top stores/categories.
2. **Three discoveries:** promotions roughly double stock-out risk; censored sales make bad reorder rules self-reinforcing; one supplier drives a category's failures.
3. **What we predict and how well:** WAPE vs baseline; recall of HIGH risk.
4. **Why trust it:** time-based validation, no leakage, calibrated probabilities, holdout week result.
5. **Action now:** top 10 store × product actions with reasons and rupees at stake.
