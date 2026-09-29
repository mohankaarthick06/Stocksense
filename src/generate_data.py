"""
StockSense - synthetic data generator for NovaMart Retail (IntelliData 2026).

Simulates the real process that produces the data, instead of sampling columns
independently:

    true demand  ->  store shelf (inventory simulation with a rule-based
    reorder policy and imperfect suppliers)  ->  sales = min(demand, stock)
    ->  individual transaction lines  ->  data-entry noise (quality traps)

Because sales are capped by stock, the recorded sales are *censored* demand,
exactly like real retail data. The true demand is written only to the answer
key so teams can prove their corrections work.

Outputs
    data/raw/                        what teams receive
        transactions.csv, products.csv, stores.csv, inventory.csv,
        external_factors.csv, promotions_calendar.csv, purchase_orders.csv,
        suppliers.csv
    data/answer_key_DO_NOT_COMMIT/   ground truth (never commit)
        true_demand_and_holdout.csv, holdout_week_summary.csv,
        reorder_policy_truth.csv, injected_issues.csv

Observed period : 2026-03-01 .. 2026-08-31
Holdout week    : 2026-09-01 .. 2026-09-07 (answer key only; calendars,
                  planned promotions and weather rows are published because
                  they are known in advance)

Run:  python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KEY = ROOT / "data" / "answer_key_DO_NOT_COMMIT"

START = pd.Timestamp("2026-03-01")
OBS_END = pd.Timestamp("2026-08-31")
HOLD_END = pd.Timestamp("2026-09-07")
RECAL_DATE = pd.Timestamp("2026-06-01")  # planners reset reorder levels from recent *sales*

DATES = pd.date_range(START, HOLD_END)
T = len(DATES)
T_OBS = (OBS_END - START).days + 1
T_RECAL = (RECAL_DATE - START).days

# --------------------------------------------------------------------------
# Master data
# --------------------------------------------------------------------------
# id: name, city, nominal lead days, P(late), delay days (choices), P(partial shipment)
SUPPLIERS = {
    "SUP01": ("Aavin Dairy Distributors", "Coimbatore", 1, 0.03, [1], 0.02),
    "SUP02": ("FreshBake Foods", "Coimbatore", 1, 0.05, [1], 0.03),
    "SUP03": ("Kaveri Staples Wholesale", "Tiruchirappalli", 3, 0.08, [1, 2], 0.03),
    "SUP04": ("FizzUp Beverages Pvt Ltd", "Chennai", 3, 0.36, [1, 2, 3, 4], 0.12),
    "SUP05": ("CrunchIt Snacks", "Madurai", 2, 0.10, [1, 2], 0.03),
    "SUP06": ("Crispo Biscuits", "Salem", 3, 0.08, [1, 2], 0.03),
    "SUP07": ("HomeGuard Distributors", "Chennai", 4, 0.10, [1, 2], 0.03),
    "SUP08": ("GlowCare Consumer Products", "Chennai", 4, 0.07, [1, 2], 0.02),
    "SUP09": ("ChillBay Frozen Foods", "Coimbatore", 2, 0.12, [1, 2], 0.04),
    "SUP10": ("PoojaShree Traders", "Madurai", 3, 0.10, [1, 2], 0.03),
}

# category: promo elasticity, NB dispersion k (lower = more volatile),
#           mean units per transaction line, planner safety days
CATEGORY = {
    "Dairy":         dict(elast=0.9, k=25, line=1.7, safety=0.3),
    "Groceries":     dict(elast=1.8, k=12, line=1.15, safety=0.7),
    "Beverages":     dict(elast=3.2, k=6,  line=2.0, safety=0.7),
    "Snacks":        dict(elast=3.8, k=5,  line=1.8, safety=0.7),
    "Personal Care": dict(elast=2.8, k=8,  line=1.1, safety=1.1),
    "Household":     dict(elast=2.4, k=8,  line=1.1, safety=1.1),
    "Frozen":        dict(elast=2.8, k=4,  line=1.3, safety=0.7),
}

# pid, name, category, sub_category, brand, mrp, cost, shelf_life, supplier,
# base units/day (reference supermarket), case_pack, temp sensitivity,
# festival sensitivity, launch date (None = on sale all period)
PRODUCTS = [
    # --- Dairy & fresh
    ("P101", "Milk 1L", "Dairy", "Milk", "Aavin", 60, 52, 3, "SUP01", 62, 10, 0.0, 0.3, None),
    ("P102", "Milk 500ml", "Dairy", "Milk", "Aavin", 30, 26, 3, "SUP01", 44, 20, 0.0, 0.3, None),
    ("P103", "Curd 500g", "Dairy", "Curd", "Aavin", 35, 27, 5, "SUP01", 24, 12, 0.3, 0.3, None),
    ("P104", "Buttermilk 200ml", "Dairy", "Buttermilk", "Aavin", 15, 10, 3, "SUP01", 20, 24, 1.0, 0.1, None),
    ("P105", "Paneer 200g", "Dairy", "Paneer", "DairyDew", 95, 70, 10, "SUP01", 7, 6, 0.0, 0.6, None),
    ("P106", "Butter 100g", "Dairy", "Butter", "DairyDew", 58, 47, 180, "SUP01", 9, 12, 0.0, 0.4, None),
    ("P107", "Ghee 500ml", "Dairy", "Ghee", "Aavin", 340, 288, 270, "SUP01", 5, 6, 0.0, 1.5, None),
    ("P108", "Cheese Slices 200g", "Dairy", "Cheese", "DairyDew", 150, 118, 180, "SUP01", 4, 6, 0.0, 0.1, None),
    ("P109", "Flavoured Milk 200ml", "Dairy", "Flavoured Milk", "Aavin", 25, 17, 90, "SUP01", 12, 24, 0.9, 0.1, None),
    ("P111", "Bread 400g", "Groceries", "Bread", "FreshBake", 45, 34, 4, "SUP02", 28, 10, 0.0, 0.1, None),
    ("P112", "Brown Bread 400g", "Groceries", "Bread", "FreshBake", 55, 41, 4, "SUP02", 8, 10, 0.0, 0.0, None),
    ("P113", "Idli Batter 1kg", "Groceries", "Batter", "BatterUp", 90, 62, 4, "SUP02", 19, 8, 0.0, 0.5, None),
    ("P114", "Greek Yogurt 100g", "Dairy", "Curd", "DairyDew", 50, 35, 15, "SUP01", 6, 12, 0.4, 0.0, "2026-08-27"),
    # --- Groceries & staples
    ("P501", "Ponni Rice 5kg", "Groceries", "Rice", "Nellai Gold", 380, 318, 365, "SUP03", 11, 4, 0.0, 0.5, None),
    ("P502", "Basmati Rice 1kg", "Groceries", "Rice", "GrainKing", 160, 124, 365, "SUP03", 5, 10, 0.0, 0.8, None),
    ("P503", "Wheat Atta 5kg", "Groceries", "Flour", "GrainKing", 290, 244, 180, "SUP03", 6, 4, 0.0, 0.3, None),
    ("P504", "Toor Dal 1kg", "Groceries", "Pulses", "GrainKing", 180, 150, 270, "SUP03", 8, 10, 0.0, 0.6, None),
    ("P505", "Sugar 1kg", "Groceries", "Sugar", "Nellai Gold", 50, 44, 365, "SUP03", 17, 20, 0.0, 0.9, None),
    ("P506", "Sunflower Oil 1L", "Groceries", "Edible Oil", "GoldDrop", 175, 150, 270, "SUP03", 13, 12, 0.0, 0.7, None),
    ("P507", "Groundnut Oil 1L", "Groceries", "Edible Oil", "Idhaya", 230, 194, 270, "SUP03", 7, 12, 0.0, 0.8, None),
    ("P508", "Salt 1kg", "Groceries", "Salt", "SeaPure", 28, 20, 730, "SUP03", 9, 25, 0.0, 0.1, None),
    ("P509", "Tea Powder 250g", "Groceries", "Tea", "TeaLeaf", 150, 118, 365, "SUP03", 8, 12, 0.0, 0.3, None),
    ("P510", "Filter Coffee 200g", "Groceries", "Coffee", "KaapiKing", 210, 168, 270, "SUP03", 7, 12, 0.0, 0.4, None),
    ("P511", "Sambar Powder 200g", "Groceries", "Spices", "SpiceNest", 75, 54, 270, "SUP03", 7, 20, 0.0, 0.4, None),
    ("P512", "Chilli Powder 200g", "Groceries", "Spices", "SpiceNest", 80, 58, 270, "SUP03", 6, 20, 0.0, 0.3, None),
    ("P513", "Rava 1kg", "Groceries", "Flour", "GrainKing", 60, 47, 180, "SUP03", 6, 10, 0.0, 0.9, None),
    ("P514", "Jaggery 1kg", "Groceries", "Sugar", "Nellai Gold", 90, 69, 240, "SUP03", 3, 10, 0.0, 1.3, None),
    ("P515", "Camphor 50g", "Groceries", "Pooja Items", "PoojaShree", 60, 38, 730, "SUP10", 3, 24, 0.0, 1.8, None),
    ("P516", "Agarbatti Pack", "Groceries", "Pooja Items", "PoojaShree", 60, 40, 730, "SUP10", 4, 24, 0.0, 1.5, None),
    ("P517", "Coconut (each)", "Groceries", "Fresh Produce", "Farm Fresh", 35, 25, 20, "SUP03", 14, 20, 0.0, 1.0, None),
    ("P518", "Instant Noodles 4-pack", "Groceries", "Instant Food", "QuickBite", 56, 44, 270, "SUP05", 14, 12, 0.0, 0.1, None),
    # --- Beverages (all from the unreliable SUP04)
    ("P330", "Cola 750ml", "Beverages", "Soft Drink", "FizzUp", 50, 31, 180, "SUP04", 24, 12, 1.0, 0.4, None),
    ("P331", "Lemon Soda 750ml", "Beverages", "Soft Drink", "FizzUp", 45, 28, 180, "SUP04", 12, 12, 1.0, 0.3, None),
    ("P332", "Orange Drink 600ml", "Beverages", "Soft Drink", "FizzUp", 40, 25, 180, "SUP04", 9, 12, 0.9, 0.3, None),
    ("P333", "Mango Drink 1.2L", "Beverages", "Juice", "Tropico", 70, 46, 180, "SUP04", 9, 12, 0.9, 0.3, None),
    ("P334", "Packaged Water 1L", "Beverages", "Water", "AquaPure", 20, 11, 365, "SUP04", 30, 12, 1.2, 0.2, None),
    ("P335", "Energy Drink 250ml", "Beverages", "Energy Drink", "Bolt", 110, 72, 365, "SUP04", 4, 24, 0.5, 0.0, None),
    ("P336", "Tender Coconut Water 200ml", "Beverages", "Juice", "Tropico", 45, 30, 60, "SUP04", 6, 24, 1.2, 0.1, None),
    ("P337", "Mixed Fruit Juice 1L", "Beverages", "Juice", "Tropico", 125, 92, 180, "SUP04", 5, 12, 0.6, 0.3, None),
    ("P338", "Cola 2L", "Beverages", "Soft Drink", "FizzUp", 95, 62, 180, "SUP04", 8, 6, 1.0, 0.6, None),
    ("P339", "Cold Coffee Can 200ml", "Beverages", "Coffee Drink", "FizzUp", 60, 38, 270, "SUP04", 8, 24, 0.8, 0.0, "2026-08-10"),
    # --- Snacks
    ("P442", "Biscuits 200g", "Snacks", "Biscuits", "Crispo", 35, 22, 240, "SUP06", 24, 24, 0.0, 0.3, None),
    ("P443", "Cream Biscuits 120g", "Snacks", "Biscuits", "Crispo", 30, 19, 240, "SUP06", 17, 24, 0.0, 0.3, None),
    ("P444", "Marie Biscuits 250g", "Snacks", "Biscuits", "Crispo", 40, 28, 240, "SUP06", 13, 24, 0.0, 0.2, None),
    ("P445", "Potato Chips 52g", "Snacks", "Chips", "CrunchIt", 20, 13, 150, "SUP05", 29, 30, 0.2, 0.4, None),
    ("P446", "Madras Mixture 200g", "Snacks", "Namkeen", "Madras Mix", 60, 40, 120, "SUP05", 10, 20, 0.0, 1.0, None),
    ("P447", "Murukku 200g", "Snacks", "Namkeen", "Madras Mix", 70, 48, 90, "SUP05", 6, 20, 0.0, 1.5, None),
    ("P448", "Chocolate Bar 50g", "Snacks", "Chocolate", "ChocoBay", 45, 30, 270, "SUP05", 14, 24, -0.2, 0.6, None),
    ("P449", "Peanut Chikki 100g", "Snacks", "Sweets", "SweetHouse", 25, 15, 120, "SUP05", 6, 30, 0.0, 0.5, None),
    ("P450", "Butter Cookies 200g", "Snacks", "Biscuits", "Crispo", 99, 66, 240, "SUP06", 5, 12, 0.0, 0.8, None),
    ("P451", "Rusk 300g", "Snacks", "Biscuits", "Crispo", 60, 42, 180, "SUP06", 6, 12, 0.0, 0.1, None),
    ("P452", "Mysore Pak Box 500g", "Snacks", "Sweets", "SweetHouse", 280, 188, 20, "SUP05", 3, 6, 0.0, 2.0, None),
    ("P453", "Millet Cookies 150g", "Snacks", "Biscuits", "Crispo", 55, 36, 240, "SUP06", 6, 24, 0.0, 0.2, "2026-08-10"),
    # --- Personal care
    ("P205", "Shampoo 180ml", "Personal Care", "Shampoo", "GlowCare", 120, 79, 730, "SUP08", 6, 12, 0.0, 0.1, None),
    ("P206", "Bath Soap 4x100g", "Personal Care", "Soap", "HerbalGlow", 160, 114, 730, "SUP08", 8, 12, 0.0, 0.2, None),
    ("P207", "Toothpaste 150g", "Personal Care", "Oral Care", "DentaFresh", 105, 72, 730, "SUP08", 10, 12, 0.0, 0.1, None),
    ("P208", "Toothbrush", "Personal Care", "Oral Care", "DentaFresh", 45, 25, 1095, "SUP08", 5, 24, 0.0, 0.1, None),
    ("P209", "Coconut Hair Oil 200ml", "Personal Care", "Hair Oil", "KeraPure", 110, 80, 540, "SUP08", 8, 12, 0.0, 0.2, None),
    ("P210", "Face Wash 100ml", "Personal Care", "Skin Care", "HerbalGlow", 150, 95, 730, "SUP08", 3, 12, 0.0, 0.1, None),
    ("P211", "Talcum Powder 100g", "Personal Care", "Skin Care", "HerbalGlow", 90, 60, 1095, "SUP08", 3, 12, 0.4, 0.1, None),
    ("P212", "Deodorant 150ml", "Personal Care", "Deodorant", "GlowCare", 199, 120, 1095, "SUP08", 2, 12, 0.3, 0.1, None),
    ("P213", "Sanitary Pads 8s", "Personal Care", "Hygiene", "GlowCare", 99, 68, 1095, "SUP08", 6, 24, 0.0, 0.0, None),
    ("P214", "Hand Wash 190ml", "Personal Care", "Hygiene", "HerbalGlow", 99, 62, 730, "SUP08", 4, 12, 0.0, 0.0, None),
    ("P216", "Herbal Shampoo 180ml", "Personal Care", "Shampoo", "GlowCare", 135, 88, 730, "SUP08", 4, 12, 0.0, 0.0, "2026-08-27"),
    # --- Household
    ("P601", "Detergent Powder 1kg", "Household", "Laundry", "WashMax", 135, 102, 730, "SUP07", 10, 12, 0.0, 0.3, None),
    ("P602", "Dishwash Bar 3-pack", "Household", "Dishwash", "ShinePro", 60, 42, 730, "SUP07", 8, 24, 0.0, 0.2, None),
    ("P603", "Dishwash Liquid 500ml", "Household", "Dishwash", "ShinePro", 110, 78, 730, "SUP07", 5, 12, 0.0, 0.2, None),
    ("P604", "Floor Cleaner 1L", "Household", "Cleaning", "ShinePro", 199, 135, 730, "SUP07", 4, 12, 0.0, 0.5, None),
    ("P605", "Toilet Cleaner 500ml", "Household", "Cleaning", "ShinePro", 99, 68, 730, "SUP07", 4, 12, 0.0, 0.2, None),
    ("P606", "Mosquito Repellent Refill", "Household", "Pest Control", "GuardHome", 85, 58, 1095, "SUP07", 5, 24, 0.0, 0.0, None),
    ("P607", "Garbage Bags 30s", "Household", "Disposables", "GuardHome", 99, 60, 1095, "SUP07", 3, 24, 0.0, 0.2, None),
    ("P608", "Matchbox 10-pack", "Household", "Pooja & Utility", "PoojaShree", 10, 6, 1095, "SUP10", 8, 50, 0.0, 0.8, None),
    ("P609", "Aluminium Foil 9m", "Household", "Disposables", "GuardHome", 120, 85, 1095, "SUP07", 2, 12, 0.0, 0.3, None),
    ("P610", "Liquid Detergent 1L", "Household", "Laundry", "WashMax", 230, 168, 730, "SUP07", 3, 6, 0.0, 0.2, None),
    # --- Frozen
    ("P701", "Vanilla Ice Cream 700ml", "Frozen", "Ice Cream", "ChillBay", 180, 118, 180, "SUP09", 6, 6, 1.4, 0.3, None),
    ("P702", "Chocolate Cone", "Frozen", "Ice Cream", "ChillBay", 40, 25, 180, "SUP09", 11, 24, 1.5, 0.2, None),
    ("P703", "Frozen Green Peas 500g", "Frozen", "Frozen Veg", "FrostFarm", 110, 78, 365, "SUP09", 4, 12, 0.0, 0.3, None),
    ("P704", "Frozen Parotta 5pc", "Frozen", "Frozen Snacks", "FrostFarm", 99, 68, 180, "SUP09", 5, 12, 0.0, 0.2, None),
    ("P705", "Chicken Nuggets 400g", "Frozen", "Frozen Snacks", "FrostFarm", 240, 168, 270, "SUP09", 3, 6, 0.0, 0.3, None),
    ("P706", "Kulfi Stick", "Frozen", "Ice Cream", "ChillBay", 30, 18, 180, "SUP09", 10, 24, 1.4, 0.2, None),
    ("P707", "French Fries 420g", "Frozen", "Frozen Snacks", "FrostFarm", 150, 100, 365, "SUP09", 3, 6, 0.0, 0.2, None),
]
PCOLS = ["product_id", "name", "category", "sub_category", "brand", "mrp", "cost_price",
         "shelf_life_days", "supplier_id", "base", "case_pack", "temp_s", "fest_s", "launch"]
prod = pd.DataFrame(PRODUCTS, columns=PCOLS)
P = len(prod)

# MRP revisions during the period: pid -> (effective date, previous MRP). products.csv holds the current MRP.
PRICE_HIKES = {
    "P330": ("2026-06-01", 45), "P506": ("2026-05-10", 160), "P101": ("2026-06-15", 56),
    "P102": ("2026-06-15", 28), "P510": ("2026-07-01", 195), "P501": ("2026-07-20", 360),
    "P509": ("2026-08-05", 140), "P452": ("2026-04-01", 260),
}

# store_id, city, store_type, floor_area, avg_daily_customers, region,
# demand scale, trend over period, weekend lift, planner bias
STORES = [
    ("S01", "Coimbatore", "Supermarket", 8500, 1250, "West", 1.00, 0.08, 1.20, 1.00),
    ("S02", "Chennai", "Hypermarket", 18000, 2850, "North", 2.55, 0.22, 1.45, 0.97),
    ("S03", "Madurai", "Express", 4200, 720, "South", 0.58, -0.18, 0.92, 1.15),
    ("S04", "Salem", "Supermarket", 7600, 1080, "Central", 0.88, 0.00, 1.18, 1.02),
    ("S05", "Tiruchirappalli", "Supermarket", 6900, 980, "Central", 0.80, 0.04, 1.22, 0.78),
    ("S06", "Chennai", "Express", 3900, 760, "North", 0.62, 0.10, 0.95, 0.95),
]
store = pd.DataFrame(STORES, columns=["store_id", "city", "store_type", "floor_area_sqft",
                                      "avg_daily_customers", "region", "scale", "trend",
                                      "weekend_lift", "planner_bias"])

AFFINITY = {
    "Express": {"Beverages": 1.3, "Snacks": 1.25, "Dairy": 1.1, "Groceries": 0.6,
                "Personal Care": 0.8, "Household": 0.7, "Frozen": 1.0},
    "Hypermarket": {"Groceries": 1.15, "Household": 1.15, "Personal Care": 1.1},
    "Supermarket": {},
}

# Festivals / holidays (Tamil Nadu calendar, 2026)
FESTIVALS = {  # date: (name, public holiday)
    "2026-03-19": ("Ugadi", 0), "2026-03-21": ("Ramzan", 1), "2026-04-14": ("Tamil New Year", 1),
    "2026-05-27": ("Bakrid", 1), "2026-08-03": ("Aadi Perukku", 0), "2026-08-26": ("Onam", 0),
    "2026-08-28": ("Varalakshmi Vratham", 0), "2026-09-04": ("Krishna Jayanthi", 1),
}
HOLIDAYS = {"2026-04-03": "Good Friday", "2026-05-01": "May Day", "2026-06-26": "Muharram",
            "2026-08-15": "Independence Day"}
MAJOR_FESTIVALS = ["2026-04-14", "2026-08-03", "2026-08-28", "2026-09-04"]

# City weather: monthly mean temperature Mar..Sep, monthly rain-day probability, mean rain mm
CITY_WEATHER = {
    "Coimbatore": ([30.5, 32.0, 31.0, 28.0, 27.2, 27.6, 28.3], [.05, .15, .25, .40, .45, .38, .30], 9),
    "Chennai": ([30.8, 33.2, 36.0, 36.2, 34.6, 33.6, 32.8], [.03, .05, .08, .15, .25, .30, .35], 12),
    "Madurai": ([33.0, 35.6, 36.8, 36.0, 35.2, 34.8, 34.0], [.05, .12, .15, .12, .12, .15, .20], 8),
    "Salem": ([31.8, 34.4, 34.8, 32.2, 31.0, 30.8, 31.0], [.05, .15, .25, .25, .30, .30, .30], 9),
    "Tiruchirappalli": ([33.2, 36.0, 37.2, 36.4, 35.4, 34.8, 34.2], [.03, .08, .12, .12, .15, .18, .22], 8),
}

# --------------------------------------------------------------------------
# Calendar and weather
# --------------------------------------------------------------------------
dstr = DATES.strftime("%Y-%m-%d")
dow = DATES.dayofweek.values
weekend = (dow >= 5).astype(int)
payday = (DATES.day <= 5).astype(int)
fest_idx = np.array([(pd.Timestamp(d) - START).days for d in FESTIVALS])
festival = np.isin(np.arange(T), fest_idx).astype(int)
holiday = np.zeros(T, int)
for d, (_, h) in FESTIVALS.items():
    holiday[(pd.Timestamp(d) - START).days] = h
for d in HOLIDAYS:
    holiday[(pd.Timestamp(d) - START).days] = 1

cities = list(CITY_WEATHER)
temp = {}
rain = {}
mid_month = np.array([(pd.Timestamp(f"2026-{m:02d}-15") - START).days for m in range(3, 10)])
for c, (tm, rp, rmm) in CITY_WEATHER.items():
    seasonal = np.interp(np.arange(T), mid_month, tm)
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.7 * ar[t - 1] + rng.normal(0, 0.9)
    month_i = DATES.month.values - 3
    wet = rng.random(T) < np.array(rp)[month_i]
    r = np.where(wet, rng.gamma(1.2, rmm, T), 0.0)
    temp[c] = np.round(seasonal + ar - 0.08 * np.minimum(r, 40), 1)
    rain[c] = np.round(r, 1)

local_event = {c: np.zeros(T, int) for c in cities}
for c in cities:
    local_event[c][rng.choice(T, size=7, replace=False)] = 1

# --------------------------------------------------------------------------
# Assortment (store x product combinations)
# --------------------------------------------------------------------------
# tier 1 = every store, tier 2 = hyper + supermarkets, tier 3 = hypermarket only
new_mask = prod["launch"].notna()
old_rank = prod.loc[~new_mask, "base"].rank(ascending=False, method="first")
tier = pd.Series(3, index=prod.index)
tier[old_rank.index[old_rank <= 46]] = 1
tier[old_rank.index[(old_rank > 46) & (old_rank <= 68)]] = 2
tier[prod["product_id"].isin(["P339", "P453"])] = 2
tier[prod["product_id"].isin(["P216", "P114"])] = 3
# a handful of festival staples are stocked everywhere regardless of rank
tier[prod["product_id"].isin(["P515", "P107"])] = 1
prod["tier"] = tier

combos = []
for si, s in store.iterrows():
    for pi, p in prod.iterrows():
        allowed = {"Hypermarket": 3, "Supermarket": 2, "Express": 1}[s.store_type]
        if p.tier <= allowed:
            combos.append((si, pi))
combos = np.array(combos)
C = len(combos)
cs, cp = combos[:, 0], combos[:, 1]
c_store = store.loc[cs].reset_index(drop=True)
c_prod = prod.loc[cp].reset_index(drop=True)
print(f"store x product combinations: {C}")

launch_t = np.array([0 if pd.isna(l) else (pd.Timestamp(l) - START).days for l in c_prod["launch"]])
active = np.arange(T)[None, :] >= launch_t[:, None]

# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
mrp_t = np.repeat(prod["mrp"].values[:, None].astype(float), T, axis=1)
for pid, (d, old) in PRICE_HIKES.items():
    pi = prod.index[prod.product_id == pid][0]
    mrp_t[pi, : (pd.Timestamp(d) - START).days] = old
everyday_disc = rng.choice([0, 0, 0.02, 0.03, 0.04], size=P)
everyday_disc[prod.product_id == "P101"] = 2 / 60  # 58 on a 60 MRP, as in the brief

# --------------------------------------------------------------------------
# Promotions calendar (planned in advance, includes the holdout week)
# --------------------------------------------------------------------------
promo = np.zeros((C, T))
promo_id_mat = np.full((C, T), "", dtype=object)
promo_rows = []
pid_counter = [0]


def add_promo(pi, t0, dur, disc, ptype, store_prob):
    stores_with = np.where(cp == pi)[0]
    chosen = stores_with[rng.random(len(stores_with)) < store_prob]
    if len(chosen) == 0:
        chosen = stores_with[:1]
    for ci in chosen:
        t_lo = max(t0, launch_t[ci])
        t_hi = min(t0 + dur, T)
        if t_hi - t_lo < 2 or promo[ci, t_lo:t_hi].any():
            continue
        pid_counter[0] += 1
        prid = f"PR{pid_counter[0]:04d}"
        promo[ci, t_lo:t_hi] = disc
        promo_id_mat[ci, t_lo:t_hi] = prid
        promo_rows.append((prid, store.store_id[cs[ci]], prod.product_id[pi], dstr[t_lo],
                           dstr[t_hi - 1], int(round(disc * 100)), ptype))


DISC_CHOICES = {"Dairy": [5, 5, 10], "Groceries": [5, 10, 10, 15], "Beverages": [10, 15, 20, 25],
                "Snacks": [10, 15, 20, 25, 30], "Personal Care": [10, 15, 20, 25],
                "Household": [10, 15, 20], "Frozen": [10, 15, 20, 25, 30]}
for pi, p in prod.iterrows():
    gap_lo, gap_hi = (40, 75) if p.shelf_life_days <= 5 else (20, 48)
    t = int(rng.integers(0, 30))
    while t < T:
        dur = int(rng.integers(4, 11))
        disc = rng.choice(DISC_CHOICES[p.category]) / 100
        add_promo(pi, t, dur, disc, "Price Off", 0.75)
        t += dur + int(rng.integers(gap_lo, gap_hi))
for fd in MAJOR_FESTIVALS:
    ft = (pd.Timestamp(fd) - START).days
    for pi, p in prod[prod.fest_s >= 0.8].iterrows():
        if rng.random() < 0.6:
            add_promo(pi, ft - 4, 5, rng.choice([10, 15, 20]) / 100, "Festival Offer", 0.8)

# --------------------------------------------------------------------------
# True demand
# --------------------------------------------------------------------------
cat = c_prod["category"].values
elast = np.array([CATEGORY[k]["elast"] for k in cat])
disp_k = np.array([CATEGORY[k]["k"] for k in cat])
city_of = c_store["city"].values
stype = c_store["store_type"].values
tt = np.arange(T)[None, :]

mu = (c_prod["base"].values * c_store["scale"].values)[:, None] * np.ones((1, T))
mu *= np.array([AFFINITY[s].get(k, 1.0) for s, k in zip(stype, cat)])[:, None]
mu *= 1 + c_store["trend"].values[:, None] * tt / T
prod_drift = rng.normal(0, 0.10, P)
mu *= np.exp(prod_drift[cp][:, None] * tt / T)

weekday_f = np.array([0.96, 0.93, 0.95, 0.97, 1.04, 1.0, 1.0])
dow_f = np.where(weekend[None, :] == 1,
                 c_store["weekend_lift"].values[:, None] * np.where(dow == 5, 1.03, 0.97)[None, :],
                 weekday_f[dow][None, :])
mu *= dow_f

hol_lift = np.array([{"Hypermarket": 1.18, "Supermarket": 1.08, "Express": 0.8}[s] for s in stype])
mu *= np.where((holiday == 1) & (festival == 0), hol_lift[:, None], 1.0)

fest_s = c_prod["fest_s"].values[:, None]
fest_f = np.ones((C, T))
for ft in fest_idx:
    for off, lift in {3: 0.25, 2: 0.45, 1: 0.70, 0: 0.30}.items():
        t = ft - off
        if 0 <= t < T:
            fest_f[:, t] *= 1 + fest_s[:, 0] * lift + (0.08 if off in (1, 2) else 0)
    if ft + 1 < T:
        fest_f[:, ft + 1] *= 1 - 0.1 * np.minimum(fest_s[:, 0], 1.5)
mu *= fest_f

pay_lift = np.where(np.isin(cat, ["Groceries", "Household", "Personal Care"]), 0.08, 0.04)
mu *= 1 + pay_lift[:, None] * payday[None, :]

temp_c = np.vstack([temp[c] for c in city_of])
rain_c = np.vstack([rain[c] for c in city_of])
mu *= np.exp(c_prod["temp_s"].values[:, None] * 0.07 * (temp_c - 31))
mu *= np.exp(-0.004 * np.minimum(rain_c, 60))
mosq = (c_prod["product_id"].values == "P606")[:, None]
mu *= np.where(mosq, 1 + 0.012 * np.minimum(rain_c, 50), 1.0)

event_c = np.vstack([local_event[c] for c in city_of])
event_lift = np.where(np.isin(cat, ["Snacks", "Beverages"]), 1.35, 1.08)
mu *= np.where(event_c == 1, event_lift[:, None], 1.0)

base_mrp = mrp_t[:, :1]
mu *= (mrp_t / base_mrp)[cp] ** -1.1

mu *= 1 + elast[:, None] * promo
long_life = (c_prod["shelf_life_days"].values >= 90)[:, None]
recent_promo = np.zeros((C, T), bool)
for lag in range(1, 8):
    recent_promo[:, lag:] |= promo[:, :-lag] > 0
mu *= np.where(long_life & recent_promo & (promo == 0), 0.85, 1.0)

since_launch = np.maximum(tt - launch_t[:, None], 0)
ramp = np.where(launch_t[:, None] > 0, np.minimum(1, 0.45 + 0.55 * since_launch / 21), 1.0)
mu *= ramp
mu = np.where(active, mu, 0.0)

k = disp_k[:, None]
demand = np.where(active, rng.negative_binomial(k, k / (k + np.maximum(mu, 1e-9))), 0)

# --------------------------------------------------------------------------
# Inventory simulation with the company's current rule-based policy
# --------------------------------------------------------------------------
sup_of = c_prod["supplier_id"].values
lead = np.array([SUPPLIERS[s][2] for s in sup_of])
late_p = np.array([SUPPLIERS[s][3] for s in sup_of])
partial_p = np.array([SUPPLIERS[s][5] for s in sup_of])
pack = c_prod["case_pack"].values
shelf = c_prod["shelf_life_days"].values
safety_days = np.array([CATEGORY[k]["safety"] for k in cat])
cycle_days = np.where(shelf <= 5, 2, np.where(shelf <= 20, 3, 7))

# Planners know a rough baseline (no growth, promotions, festivals or heat)
planner_rate = (c_prod["base"].values * c_store["scale"].values
                * np.array([AFFINITY[s].get(k, 1.0) for s, k in zip(stype, cat)])
                * c_store["planner_bias"].values * rng.lognormal(0, 0.12, C))
planner_rate = np.where(launch_t > 0, planner_rate * 0.7, planner_rate)
rate_before = planner_rate.copy()


def policy(rate):
    rl = np.ceil(rate * (lead + safety_days))
    out = rl + np.ceil(rate * cycle_days)
    return rl, out


reorder_level, order_up_to = policy(planner_rate)

opening = np.zeros((C, T), int)
received = np.zeros((C, T), int)
sold = np.zeros((C, T), int)
closing = np.zeros((C, T), int)
rl_hist = np.zeros((C, T), int)
arrivals = np.zeros((C, T + 20), int)
on_order = np.zeros(C, int)
stock = np.zeros(C, int)
pos_log = []  # purchase orders

for t in range(T):
    starting = launch_t == t
    stock[starting] = (reorder_level[starting]
                       + np.ceil(rng.uniform(0.3, 1.0, starting.sum()) * (order_up_to - reorder_level)[starting]))
    if t == T_RECAL:
        # planners recalibrate from the last 30 days of *sales* (censored by stock-outs)
        hist = sold[:, t - 30:t].astype(float)
        days = active[:, t - 30:t].sum(1)
        new_rate = np.where(days >= 14, hist.sum(1) / np.maximum(days, 1), planner_rate)
        planner_rate = np.maximum(new_rate, 0.3)
        reorder_level, order_up_to = policy(planner_rate)
    if t == T_OBS:
        snap_stock = stock.copy()
        snap_arrivals = arrivals.copy()
    rec = arrivals[:, t].copy()
    on_order -= rec
    opening[:, t] = stock
    received[:, t] = rec
    avail = stock + rec
    s_ = np.minimum(demand[:, t], avail)
    sold[:, t] = s_
    closing[:, t] = avail - s_
    stock = avail - s_
    rl_hist[:, t] = reorder_level

    position = stock + on_order
    need = active[:, t] & (position <= reorder_level)
    for ci in np.where(need)[0]:
        qty = int(np.ceil((order_up_to[ci] - position[ci]) / pack[ci]) * pack[ci])
        if qty <= 0:
            continue
        s_id = sup_of[ci]
        delay = int(rng.choice(SUPPLIERS[s_id][4])) if rng.random() < late_p[ci] else 0
        q_rec = qty
        if rng.random() < partial_p[ci]:
            q_rec = int(np.floor(qty * rng.uniform(0.5, 0.9) / pack[ci]) * pack[ci])
        arr = t + lead[ci] + delay
        arrivals[ci, arr] += q_rec
        on_order[ci] += q_rec
        pos_log.append((ci, t, t + lead[ci], arr, qty, q_rec))

stockout = (closing == 0) & active

# Holdout counterfactual: no new orders after 31 Aug, only already-open POs arrive
na_sold = np.zeros((C, T - T_OBS), int)
na_close = np.zeros((C, T - T_OBS), int)
stk = snap_stock.copy()
for j, t in enumerate(range(T_OBS, T)):
    avail = stk + snap_arrivals[:, t]
    na_sold[:, j] = np.minimum(demand[:, t], avail)
    stk = avail - na_sold[:, j]
    na_close[:, j] = stk

# --------------------------------------------------------------------------
# Transactions (observed period only), built from units actually sold
# --------------------------------------------------------------------------
obs = np.arange(T_OBS)
rows_c, rows_t = np.nonzero(sold[:, :T_OBS] > 0)
S = sold[rows_c, rows_t]
line_mean = np.array([CATEGORY[k]["line"] for k in cat])
line_mean = np.where(c_prod["sub_category"].values == "Water", 2.3, line_mean)
line_mean = np.where(c_prod["sub_category"].values == "Milk", 1.8, line_mean)
n_lines = np.clip(np.round(S / line_mean[rows_c] * rng.uniform(0.85, 1.15, len(S))), 1, S).astype(int)
offsets = np.concatenate([[0], np.cumsum(n_lines)[:-1]])
L = n_lines.sum()
extra = S - n_lines
unit_row = np.repeat(np.arange(len(S)), extra)
pos_in_row = np.floor(rng.random(len(unit_row)) * n_lines[unit_row]).astype(int)
# genuine bulk purchases (caterers, small shops): a line absorbs most extra units
bulk_rows = (S >= 40) & (rng.random(len(S)) < 0.04)
to_bulk = bulk_rows[unit_row] & (rng.random(len(unit_row)) < 0.6)
pos_in_row[to_bulk] = 0
line_qty = 1 + np.bincount(offsets[unit_row] + pos_in_row, minlength=L)
line_c = np.repeat(rows_c, n_lines)
line_t = np.repeat(rows_t, n_lines)

disc_line = promo[line_c, line_t]
mrp_line = mrp_t[cp[line_c], line_t]
hyper_extra = np.where(stype[line_c] == "Hypermarket", 0.01, 0.0)
price_line = np.where(disc_line > 0, mrp_line * (1 - disc_line),
                      mrp_line * (1 - everyday_disc[cp[line_c]] - hyper_extra))
price_line = np.where(mrp_line >= 20, np.round(price_line), np.round(price_line, 2))
price_line = np.minimum(price_line, mrp_line)

HOUR_PROFILE = {
    "Supermarket": {7: 2, 8: 4, 9: 6, 10: 9, 11: 9, 12: 6, 13: 5, 14: 4, 15: 4, 16: 6, 17: 8, 18: 10, 19: 11, 20: 9, 21: 5, 22: 2},
    "Hypermarket": {9: 3, 10: 5, 11: 7, 12: 7, 13: 6, 14: 6, 15: 6, 16: 7, 17: 9, 18: 11, 19: 12, 20: 11, 21: 7, 22: 3},
    "Express": {6: 4, 7: 9, 8: 11, 9: 8, 10: 5, 11: 4, 12: 4, 13: 4, 14: 3, 15: 3, 16: 4, 17: 7, 18: 10, 19: 10, 20: 8, 21: 5, 22: 3},
}
PAY = {"Hypermarket": [0.50, 0.32, 0.18], "Supermarket": [0.55, 0.20, 0.25], "Express": [0.60, 0.08, 0.32]}
line_stype = stype[line_c]
hour = np.zeros(L, int)
pay = np.empty(L, dtype=object)
for st, prof in HOUR_PROFILE.items():
    m = line_stype == st
    hrs = np.array(list(prof))
    w = np.array(list(prof.values()), float)
    hour[m] = rng.choice(hrs, size=m.sum(), p=w / w.sum())
    pay[m] = rng.choice(["UPI", "Card", "Cash"], size=m.sum(), p=PAY[st])

store_num = cs[line_c] + 1
pool = c_store["avg_daily_customers"].values[line_c] * 4
cust_idx = np.floor(pool * rng.random(L) ** 2).astype(int) + 1
cust = np.char.add(np.char.add("C", np.char.zfill(store_num.astype(str), 2)),
                   np.char.zfill(cust_idx.astype(str), 5)).astype(object)
cust[rng.random(L) < 0.38] = ""  # not a loyalty member

tx = pd.DataFrame({
    "date": dstr[line_t], "store_id": c_store["store_id"].values[line_c],
    "product_id": c_prod["product_id"].values[line_c], "quantity": line_qty,
    "selling_price": price_line, "discount_pct": np.round(disc_line * 100).astype(int),
    "promotion_flag": (disc_line > 0).astype(int), "customer_id": cust,
    "payment_mode": pay, "hour": hour, "_t": line_t,
})
tx = tx.sample(frac=1, random_state=SEED).sort_values(["_t", "hour"], kind="stable").drop(columns="_t")
tx.insert(0, "transaction_id", [f"T{i:07d}" for i in range(1, len(tx) + 1)])
tx = tx.reset_index(drop=True)

# --------------------------------------------------------------------------
# Clean tables
# --------------------------------------------------------------------------
inv_c, inv_t = np.nonzero(active[:, :T_OBS])
inv = pd.DataFrame({
    "date": dstr[inv_t], "store_id": c_store["store_id"].values[inv_c],
    "product_id": c_prod["product_id"].values[inv_c], "opening_stock": opening[inv_c, inv_t],
    "received_qty": received[inv_c, inv_t], "sold_qty": sold[inv_c, inv_t],
    "closing_stock": closing[inv_c, inv_t], "reorder_level": rl_hist[inv_c, inv_t],
    "lead_time_days": lead[inv_c],
}).sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

products_out = prod[["product_id", "category", "sub_category", "brand", "mrp", "cost_price",
                     "shelf_life_days", "supplier_id"]].copy()
products_out.insert(1, "product_name", prod["name"])

stores_out = store[["store_id", "city", "store_type", "floor_area_sqft", "avg_daily_customers", "region"]].copy()

ext = []
for c in cities:
    ext.append(pd.DataFrame({
        "date": dstr, "city": c, "temp_c": temp[c], "rain_mm": rain[c], "holiday": holiday,
        "festival": festival, "festival_name": [FESTIVALS.get(d, ("",))[0] for d in dstr],
        "weekend": weekend, "local_event": local_event[c],
    }))
ext = pd.concat(ext).sort_values(["date", "city"]).reset_index(drop=True)

promos_out = pd.DataFrame(promo_rows, columns=["promo_id", "store_id", "product_id", "start_date",
                                               "end_date", "discount_pct", "promo_type"])
promos_out = promos_out.sort_values(["start_date", "store_id", "product_id"]).reset_index(drop=True)

po = pd.DataFrame(pos_log, columns=["ci", "t_order", "t_exp", "t_arr", "qty_ordered", "qty_rec"])
po = po[po.t_order < T_OBS].sort_values(["t_order", "ci"]).reset_index(drop=True)
is_open = po.t_arr >= T_OBS
po_out = pd.DataFrame({
    "po_id": [f"PO{i:06d}" for i in range(1, len(po) + 1)],
    "store_id": c_store["store_id"].values[po.ci], "product_id": c_prod["product_id"].values[po.ci],
    "supplier_id": sup_of[po.ci], "order_date": dstr[po.t_order], "expected_date": dstr[po.t_exp],
    "received_date": np.where(is_open, "", dstr[np.minimum(po.t_arr, T - 1)]),
    "qty_ordered": po.qty_ordered, "qty_received": np.where(is_open, "", po.qty_rec.astype(str)),
    "status": np.where(is_open, "OPEN", "RECEIVED"),
})

suppliers_out = pd.DataFrame([(k, v[0], v[1], v[2]) for k, v in SUPPLIERS.items()],
                             columns=["supplier_id", "supplier_name", "city", "nominal_lead_days"])

# sanity: transactions reconcile exactly with inventory before any noise is added
chk = tx.groupby(["date", "store_id", "product_id"]).quantity.sum()
inv_chk = inv.set_index(["date", "store_id", "product_id"]).sold_qty
inv_chk = inv_chk[inv_chk > 0]
assert chk.sort_index().equals(inv_chk.sort_index().rename("quantity")), "tx/inventory mismatch"
assert (inv.opening_stock + inv.received_qty - inv.sold_qty == inv.closing_stock).all()

# --------------------------------------------------------------------------
# Data quality traps
# --------------------------------------------------------------------------
issues = []


def log(file, issue, n, note):
    issues.append((file, issue, int(n), note))


def pick(n, pool_idx):
    return rng.choice(pool_idx, size=n, replace=False)


N = len(tx)
free = np.ones(N, bool)


def take(n, mask=None):
    cand = np.where(free & (mask if mask is not None else True))[0]
    idx = pick(n, cand)
    free[idx] = False
    return idx


i_key = take(45, (tx.quantity.values <= 5))
tx.loc[i_key, "quantity"] = tx.loc[i_key, "quantity"].values * 100
log("transactions.csv", "keying error: quantity x100 (inventory.sold_qty disagrees)", len(i_key),
    "original quantity = value / 100")

i_nan = take(int(N * 0.005))
tx["selling_price"] = tx["selling_price"].astype(float)
tx.loc[i_nan, "selling_price"] = np.nan
log("transactions.csv", "missing selling_price", len(i_nan), "")

i_10x = take(int(N * 0.001))
tx.loc[i_10x, "selling_price"] = tx.loc[i_10x, "selling_price"] * 10
log("transactions.csv", "selling_price ~10x MRP (decimal slip)", len(i_10x), "")

i_date = take(int(N * 0.012))
tx.loc[i_date, "date"] = pd.to_datetime(tx.loc[i_date, "date"]).dt.strftime("%d-%m-%Y")
log("transactions.csv", "date in DD-MM-YYYY format", len(i_date), "rest are YYYY-MM-DD")

i_sid = take(int(N * 0.004))
tx.loc[i_sid, "store_id"] = [rng.choice([s.lower(), f" {s}", f"{s} "]) for s in tx.loc[i_sid, "store_id"]]
i_pid = take(int(N * 0.004))
tx.loc[i_pid, "product_id"] = [rng.choice([p.lower(), f" {p}", f"{p} "]) for p in tx.loc[i_pid, "product_id"]]
log("transactions.csv", "store_id with lower case / whitespace", len(i_sid), "")
log("transactions.csv", "product_id with lower case / whitespace", len(i_pid), "")

n_ret = int(N * 0.003)
ret = tx.loc[pick(n_ret, np.arange(N))].copy()
ret["quantity"] = rng.choice([-1, -1, -2, -3, -4, 0], size=n_ret)
ret["transaction_id"] = [f"T{i:07d}" for i in range(N + 1, N + n_ret + 1)]
tx = pd.concat([tx, ret], ignore_index=True)
log("transactions.csv", "quantity <= 0 (returns / voids, not in inventory.sold_qty)", n_ret, "")

n_dup = int(len(tx) * 0.008)
dups = tx.loc[pick(n_dup, np.arange(len(tx)))]
tx = pd.concat([tx, dups], ignore_index=True)
tx = tx.sort_values("transaction_id", kind="stable").reset_index(drop=True)
log("transactions.csv", "exact duplicate rows", n_dup, "same transaction_id and content")
log("transactions.csv", "blank customer_id", int((tx.customer_id == "").sum()),
    "non-loyalty customers - legitimately blank")

# products
CAT_VARIANTS = {"Beverages": ["beverage", "BEVERAGES", "Beverages "], "Groceries": ["Grocery", "groceries"],
                "Snacks": ["snacks", "SNACKS"], "Personal Care": ["Personal care", "PERSONAL CARE"],
                "Dairy": ["dairy ", "DAIRY"], "Household": ["household"], "Frozen": ["FROZEN"]}
n_cat = 0
for catn, variants in CAT_VARIANTS.items():
    idx = products_out.index[products_out.category == catn]
    chosen = rng.choice(idx, size=min(len(idx), len(variants) + 1), replace=False)
    for i, v in zip(chosen, variants * 2):
        products_out.loc[i, "category"] = v
        n_cat += 1
log("products.csv", "inconsistent category spelling / case / whitespace", n_cat, "")
products_out["shelf_life_days"] = products_out["shelf_life_days"].astype("Int64")
products_out.loc[products_out.product_id == "P443", "shelf_life_days"] = pd.NA
products_out.loc[products_out.product_id == "P609", "brand"] = ""
log("products.csv", "missing shelf_life_days", 1, "P443 - peers in same sub_category have 240")
log("products.csv", "missing brand", 1, "P609")
pdup = products_out.loc[rng.choice(products_out.index, 3, replace=False)]
products_out = pd.concat([products_out, pdup]).sort_values("product_id", kind="stable").reset_index(drop=True)
log("products.csv", "duplicate product rows", 3, "")

# stores
stores_out.loc[stores_out.store_id == "S02", "store_type"] = "hypermarket"
stores_out.loc[stores_out.store_id == "S06", "store_type"] = "Express "
stores_out.loc[stores_out.store_id == "S04", "store_type"] = "SuperMarket"
log("stores.csv", "store_type case / whitespace", 3, "S02, S04, S06")

# inventory
NI = len(inv)
free_i = np.ones(NI, bool)
last_day = inv.groupby(["store_id", "product_id"]).date.transform("max") == inv.date
first_day = inv.groupby(["store_id", "product_id"]).date.transform("min") == inv.date


def take_i(n, mask):
    idx = pick(n, np.where(free_i & mask)[0])
    free_i[idx] = False
    return idx


j = take_i(int(NI * 0.010), ~last_day.values)
delta = rng.choice(np.r_[-25:0, 1:26], size=len(j))
inv.loc[j, "closing_stock"] = np.maximum(inv.loc[j, "closing_stock"].values + delta, 0)
log("inventory.csv", "closing_stock miscount (next day's opening_stock is the true count)", len(j), "")

j = take_i(int(NI * 0.003), (inv.received_qty > 0).values & ~last_day.values)
inv.loc[j, "received_qty"] = 0
log("inventory.csv", "delivery not recorded: received_qty = 0 (closing is correct)", len(j), "")

j = take_i(40, ~last_day.values)
inv.loc[j, "closing_stock"] = -rng.integers(1, 21, len(j))
log("inventory.csv", "negative closing_stock", len(j), "")

j = take_i(int(NI * 0.004), ~first_day.values)
inv = inv.drop(index=j)
log("inventory.csv", "missing rows (store-product-day absent)", len(j), "")
idup = inv.loc[rng.choice(inv.index, int(NI * 0.003), replace=False)]
inv = pd.concat([inv, idup]).sort_values(["date", "store_id", "product_id"], kind="stable").reset_index(drop=True)
log("inventory.csv", "exact duplicate rows", len(idup), "")

# external factors
ext.loc[ext.city == "Tiruchirappalli", "city"] = "Trichy"
log("external_factors.csv", "city alias 'Trichy' (stores.csv says Tiruchirappalli)", (ext.city == "Trichy").sum(), "")
NE = len(ext)
e_nan_t = rng.choice(NE, int(NE * 0.03), replace=False)
ext.loc[e_nan_t, "temp_c"] = np.nan
rest = np.setdiff1d(np.arange(NE), e_nan_t)
e_nan_r = rng.choice(rest, int(NE * 0.02), replace=False)
ext.loc[e_nan_r, "rain_mm"] = np.nan
rest = np.setdiff1d(rest, e_nan_r)
e_bad = rng.choice(rest, 12, replace=False)
ext.loc[e_bad, "temp_c"] = rng.choice([0.0, -1.0, 99.9, 58.4, 3.2, 112.0], 12)
log("external_factors.csv", "missing temp_c", len(e_nan_t), "")
log("external_factors.csv", "missing rain_mm", len(e_nan_r), "")
log("external_factors.csv", "impossible temp_c (sensor error)", len(e_bad), "valid range ~10-45")
edup = ext.loc[rng.choice(NE, 8, replace=False)]
ext = pd.concat([ext, edup]).sort_values(["date", "city"], kind="stable").reset_index(drop=True)
log("external_factors.csv", "exact duplicate rows", 8, "")

log("products.csv", "sparse history: products launched late", 4,
    "P339, P453 launched 2026-08-10; P114, P216 launched 2026-08-27 (5 days of data)")

# --------------------------------------------------------------------------
# Answer key
# --------------------------------------------------------------------------
ak_c, ak_t = np.nonzero(active)
hold = ak_t >= T_OBS
jj = np.where(hold, ak_t - T_OBS, 0)
ak = pd.DataFrame({
    "date": dstr[ak_t], "store_id": c_store["store_id"].values[ak_c],
    "product_id": c_prod["product_id"].values[ak_c], "true_demand": demand[ak_c, ak_t],
    "expected_demand": np.round(mu[ak_c, ak_t], 2), "is_holdout": hold.astype(int),
    "sold_bau": sold[ak_c, ak_t], "closing_bau": closing[ak_c, ak_t],
    "stockout_bau": stockout[ak_c, ak_t].astype(int),
    "lost_units_bau": demand[ak_c, ak_t] - sold[ak_c, ak_t],
    "sold_if_no_action": np.where(hold, na_sold[ak_c, jj], np.nan),
    "closing_if_no_action": np.where(hold, na_close[ak_c, jj], np.nan),
    "lost_units_if_no_action": np.where(hold, demand[ak_c, ak_t] - na_sold[ak_c, jj], np.nan),
}).sort_values(["date", "store_id", "product_id"])

hw = slice(T_OBS, T)
price_hold = mrp_t[cp, T_OBS] * (1 - everyday_disc[cp])
summary = pd.DataFrame({
    "store_id": c_store["store_id"], "product_id": c_prod["product_id"],
    "origin_date": OBS_END.strftime("%Y-%m-%d"),
    "closing_stock_aug31": closing[:, T_OBS - 1],
    "true_demand_7d": demand[:, hw].sum(1),
    "stockout_any_7d_bau": stockout[:, hw].any(1).astype(int),
    "lost_units_7d_bau": (demand[:, hw] - sold[:, hw]).sum(1),
    "stockout_any_7d_if_no_action": (na_close == 0).any(1).astype(int),
    "lost_units_7d_if_no_action": (demand[:, hw] - na_sold).sum(1),
    "unit_price": np.round(price_hold, 2),
})
summary["lost_revenue_7d_if_no_action"] = np.round(summary.lost_units_7d_if_no_action * summary.unit_price, 2)

policy_truth = pd.DataFrame({
    "store_id": c_store["store_id"], "product_id": c_prod["product_id"],
    "planner_rate_before_jun1": np.round(rate_before, 2), "planner_rate_after_jun1": np.round(planner_rate, 2),
    "true_mean_demand_may": np.round(demand[:, T_RECAL - 30:T_RECAL].mean(1), 2),
    "mean_sold_may": np.round(sold[:, T_RECAL - 30:T_RECAL].mean(1), 2),
    "stockout_days_may": stockout[:, T_RECAL - 30:T_RECAL].sum(1),
})

# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------
RAW.mkdir(parents=True, exist_ok=True)
KEY.mkdir(parents=True, exist_ok=True)
tx.to_csv(RAW / "transactions.csv", index=False)
products_out.to_csv(RAW / "products.csv", index=False)
stores_out.to_csv(RAW / "stores.csv", index=False)
inv.to_csv(RAW / "inventory.csv", index=False)
ext.to_csv(RAW / "external_factors.csv", index=False)
promos_out.to_csv(RAW / "promotions_calendar.csv", index=False)
po_out.to_csv(RAW / "purchase_orders.csv", index=False)
suppliers_out.to_csv(RAW / "suppliers.csv", index=False)
ak.to_csv(KEY / "true_demand_and_holdout.csv", index=False)
summary.to_csv(KEY / "holdout_week_summary.csv", index=False)
policy_truth.to_csv(KEY / "reorder_policy_truth.csv", index=False)
pd.DataFrame(issues, columns=["file", "issue", "count", "note"]).to_csv(KEY / "injected_issues.csv", index=False)

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
so = stockout[:, :T_OBS][active[:, :T_OBS]]
fwd = np.zeros((C, T_OBS), bool)
for h in range(1, 8):
    fwd[:, : T_OBS - h] |= stockout[:, h:T_OBS]
valid = active[:, :T_OBS].copy()
valid[:, T_OBS - 7:] = False
milk_s02 = np.where((c_store.store_id == "S02") & (c_prod.product_id == "P101"))[0][0]
promo_obs = promo[:, :T_OBS][active[:, :T_OBS]] > 0
p1, p0 = so[promo_obs].mean(), so[~promo_obs].mean()
po_rec = po[~is_open]
late = po_rec.t_arr > po_rec.t_exp
print(f"transaction lines          : {len(tx):,}")
print(f"inventory rows             : {len(inv):,}")
print(f"units sold / true demand   : {sold[:, :T_OBS].sum():,} / {demand[:, :T_OBS].sum():,}"
      f"  (lost {1 - sold[:, :T_OBS].sum() / demand[:, :T_OBS].sum():.1%})")
print(f"daily stock-out rate       : {so.mean():.1%}")
print(f"stock-out in next 7 days   : {fwd[valid].mean():.1%}")
print(f"stock-out rate promo/non   : {p1:.1%} / {p0:.1%}")
print(f"S02 milk 1L mean sold/day  : {sold[milk_s02, :T_OBS].mean():.0f}")
print(f"purchase orders            : {len(po_out):,} ({is_open.sum()} open)")
for s_id in SUPPLIERS:
    m = sup_of[po_rec.ci] == s_id
    print(f"   {s_id} late rate          : {late[m].mean():.1%}")
print("stock-out rate by store    :", {store.store_id[i]: round(float(stockout[cs == i, :T_OBS][active[cs == i, :T_OBS]].mean()), 3)
                                     for i in range(len(store))})
print("stock-out rate by category :", {k: round(float(stockout[cat == k, :T_OBS][active[cat == k, :T_OBS]].mean()), 3)
                                     for k in CATEGORY})
print(f"holdout: stock-outs BAU {summary.stockout_any_7d_bau.mean():.1%}, "
      f"no action {summary.stockout_any_7d_if_no_action.mean():.1%}, "
      f"lost revenue if no action Rs {summary.lost_revenue_7d_if_no_action.sum():,.0f}")
