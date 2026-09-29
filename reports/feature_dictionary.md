# Feature dictionary - Model 1 (demand) and Model 2 (stock-out)

One row = one forecast origin: close of day *t* for one store x product. Both models predict *t+1..t+7*.

**Leakage rule.** Every feature uses only information available at close of day *t*. `check_no_leakage()` in `src/features.py` rebuilds the features from data cut off at *t* (sales, stock, weather and PO arrivals after *t* removed) and asserts they are identical.

Missing values are left as NaN on purpose: they occur only where there is no history yet (new products, the first days of the period). XGBoost handles NaN natively; the other models impute.

| Group | Why it matters | Information used |
|---|---|---|
| Recent demand | Level, weekly pattern, momentum and volatility. Built on demand_est (sales corrected for empty shelves). | Days <= t |
| Promotion plan | From the promotions calendar (planned in advance), never from future transactions. Captures lift and the post-promotion dip. | Planned calendar for t+1..t+7; history <= t |
| Festival & holiday calendar | Festivals drive pre-festival buying (pooja items, ghee, sweets). Calendars are published in advance. | Published calendar; festival lift from festivals already past at t |
| Weather | Heat drives beverages and ice cream. Past 3 days only; future weather would be a forecast. | Days <= t |
| Price | Several MRP hikes happen mid-period; demand falls after a hike. | Days <= t |
| Stock & supply | What is on the shelf and on the way (open POs by expected date). Mainly for Model 2; supplier reliability is an expanding rate over POs already received. | Closing stock of t; POs ordered <= t (expected date) |
| Store & product profile | Lets one global model share learning across 387 combinations and handle new products. | Static |
| Calendar position | Where the forecast window sits in the week / season. | Known |

## Every feature

| Group | Feature | Missing |
|---|---|---|
| Recent demand | `lag_1` | 0.0% |
| Recent demand | `lag_7` | 3.3% |
| Recent demand | `lag_14` | 7.2% |
| Recent demand | `rolling_mean_7` | 0.0% |
| Recent demand | `rolling_mean_14` | 0.0% |
| Recent demand | `rolling_mean_28` | 0.0% |
| Recent demand | `rolling_std_7` | 0.6% |
| Recent demand | `rolling_std_28` | 0.6% |
| Recent demand | `cv_28` | 0.6% |
| Recent demand | `prev_week_mean` | 3.9% |
| Recent demand | `trend_7_28` | 0.0% |
| Recent demand | `growth_wow` | 3.9% |
| Recent demand | `baseline_7d` | 0.0% |
| Promotion plan | `promo_days_next7` | 0.0% |
| Promotion plan | `max_discount_next7` | 0.0% |
| Promotion plan | `avg_discount_next7` | 0.0% |
| Promotion plan | `promo_today` | 0.0% |
| Promotion plan | `promo_days_last7` | 0.0% |
| Promotion plan | `days_since_promo` | 0.0% |
| Promotion plan | `post_promo_window` | 0.0% |
| Promotion plan | `discount_today` | 0.0% |
| Festival & holiday calendar | `holiday_next7` | 0.0% |
| Festival & holiday calendar | `festival_next7` | 0.0% |
| Festival & holiday calendar | `days_to_festival` | 0.0% |
| Festival & holiday calendar | `product_festival_lift` | 0.0% |
| Festival & holiday calendar | `festival_x_lift` | 0.0% |
| Festival & holiday calendar | `local_event_next7` | 0.0% |
| Festival & holiday calendar | `payday_days_next7` | 0.0% |
| Weather | `temp_last3` | 0.0% |
| Weather | `rain_last3` | 0.0% |
| Weather | `temp_anomaly` | 0.0% |
| Price | `regular_price` | 0.0% |
| Price | `price_ratio_28` | 0.0% |
| Price | `price_change_30d` | 0.0% |
| Price | `mrp` | 0.0% |
| Stock & supply | `closing_stock` | 0.0% |
| Stock & supply | `reorder_level` | 0.0% |
| Stock & supply | `reorder_gap` | 0.0% |
| Stock & supply | `incoming_next7` | 0.0% |
| Stock & supply | `days_of_cover` | 0.0% |
| Stock & supply | `stock_to_demand_7d` | 0.0% |
| Stock & supply | `stockout_today` | 0.0% |
| Stock & supply | `stockout_days_last14` | 0.0% |
| Stock & supply | `stockout_rate_last28` | 0.0% |
| Stock & supply | `days_since_stockout` | 0.0% |
| Stock & supply | `lost_units_last7` | 0.0% |
| Stock & supply | `lead_time_days` | 0.0% |
| Stock & supply | `supplier_late_rate` | 1.5% |
| Stock & supply | `supplier_avg_delay` | 1.5% |
| Stock & supply | `supplier_fill_rate` | 1.5% |
| Stock & supply | `effective_lead_time` | 0.0% |
| Store & product profile | `store_code` | 0.0% |
| Store & product profile | `store_type` | 0.0% |
| Store & product profile | `category` | 0.0% |
| Store & product profile | `sub_category` | 0.0% |
| Store & product profile | `shelf_life_days` | 0.0% |
| Store & product profile | `avg_daily_customers` | 0.0% |
| Store & product profile | `floor_area_sqft` | 0.0% |
| Store & product profile | `history_days` | 0.0% |
| Calendar position | `origin_dow` | 0.0% |
| Calendar position | `month` | 0.0% |

## Targets

| Target | Definition | Used by |
|---|---|---|
| `next_7_day_demand` | Sum of `demand_est` over t+1..t+7 | Model 1 (Student 2) |
| `next_7_day_units_sold` | Sum of `units_sold` over t+1..t+7 (censored by stock-outs) | Comparison only |
| `stockout_next7` | 1 if closing stock hits 0 on any day in t+1..t+7 | Model 2 (Student 3) |
| `stockout_days_next7`, `lost_units_next7` | Count / lost units in the window | Model 2 analysis, prioritisation |

Dropped on purpose: `weekend_days_next7` - every 7-day window contains exactly 2 weekend days, so it is constant.