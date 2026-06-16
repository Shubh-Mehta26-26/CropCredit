import os
import pandas as pd
import numpy as np

np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. RICE CLEANING ─────────────────────────────────────────────────────────
rice_raw_path = os.path.join(BASE_DIR, "rice.csv")
df_rice = pd.read_csv(rice_raw_path, skiprows=1)
df_rice.columns = df_rice.columns.str.strip()
df_rice = df_rice.rename(columns={
    "Arrival Date": "date",
    "Arrivals (Metric Tonnes)": "market_arrivals",
    "Modal Price (Rs./Quintal)": "current_price",
    "Minimum Price (Rs./Quintal)": "min_price",
    "Maximum Price (Rs./Quintal)": "max_price",
    "Variety": "commodity"
})
# Forward fill to preserve entries with multiple varieties under one date/arrival record
df_rice["date"] = df_rice["date"].ffill()
df_rice["market_arrivals"] = df_rice["market_arrivals"].ffill()

df_rice["date"] = pd.to_datetime(df_rice["date"], errors="coerce")
cols = ["market_arrivals", "current_price", "min_price", "max_price"]
for col in cols:
    df_rice[col] = pd.to_numeric(df_rice[col], errors="coerce")
df_rice = df_rice.dropna(subset=["current_price", "market_arrivals"])
df_rice = df_rice[(df_rice["market_arrivals"] > 0) & (df_rice["current_price"] > 0)]

df_rice["commodity"] = "Rice"
df_rice["commodity_encoded"] = 2  # Global encoding index for Rice

# Generate realistic grain warehouse features for Rice
n_rice = len(df_rice)
df_rice["tonnage"] = np.random.uniform(100, 3000, n_rice)
df_rice["moisture_content"] = np.random.uniform(10.0, 15.5, n_rice)
df_rice["warehouse_temp"] = np.random.uniform(15.0, 32.0, n_rice)
df_rice["humidity"] = np.random.uniform(50.0, 80.0, n_rice)
df_rice["rainfall_deficit"] = np.random.normal(10.0, 20.0, n_rice)
df_rice["future_price"] = df_rice["current_price"] * (1 + 0.04 * np.random.randn(n_rice))

# Balanced spoilage labeling rules for Rice:
# High (1) if moisture > 14.0%, OR (moisture > 13.0% AND temp > 28°C AND humidity > 65%), OR (temp > 31.0% AND humidity > 75%)
df_rice["spoilage_label"] = (
    (df_rice["moisture_content"] > 14.0) |
    ((df_rice["moisture_content"] > 13.0) & (df_rice["warehouse_temp"] > 28.0) & (df_rice["humidity"] > 65.0)) |
    ((df_rice["warehouse_temp"] > 31.0) & (df_rice["humidity"] > 75.0))
).astype(int)

# ── 2. WHEAT CLEANING ────────────────────────────────────────────────────────
wheat_raw_path = os.path.join(BASE_DIR, "wheat.csv")
df_wheat = pd.read_csv(wheat_raw_path)
df_wheat.columns = df_wheat.columns.str.strip()
df_wheat = df_wheat.rename(columns={
    "Arrival Date": "date",
    "Arrivals (Metric Tonnes)": "market_arrivals",
    "Modal Price (Rs./Quintal)": "current_price",
    "Minimum Price (Rs./Quintal)": "min_price",
    "Maximum Price (Rs./Quintal)": "max_price",
    "Variety": "commodity"
})
df_wheat["date"] = df_wheat["date"].ffill()
df_wheat["market_arrivals"] = df_wheat["market_arrivals"].ffill()

df_wheat["date"] = pd.to_datetime(df_wheat["date"], errors="coerce")
for col in cols:
    df_wheat[col] = pd.to_numeric(df_wheat[col], errors="coerce")
df_wheat = df_wheat.dropna(subset=["current_price", "market_arrivals"])
df_wheat = df_wheat[(df_wheat["market_arrivals"] > 0) & (df_wheat["current_price"] > 0)]

df_wheat["commodity"] = "Wheat"
df_wheat["commodity_encoded"] = 4  # Global encoding index for Wheat

# Generate realistic grain warehouse features for Wheat
n_wheat = len(df_wheat)
df_wheat["tonnage"] = np.random.uniform(100, 3000, n_wheat)
df_wheat["moisture_content"] = np.random.uniform(9.0, 15.0, n_wheat)
df_wheat["warehouse_temp"] = np.random.uniform(15.0, 32.0, n_wheat)
df_wheat["humidity"] = np.random.uniform(45.0, 75.0, n_wheat)
df_wheat["rainfall_deficit"] = np.random.normal(10.0, 20.0, n_wheat)
df_wheat["future_price"] = df_wheat["current_price"] * (1 + 0.04 * np.random.randn(n_wheat))

# Balanced spoilage labeling rules for Wheat:
# High (1) if moisture > 13.5%, OR (moisture > 12.5% AND temp > 28.0% AND humidity > 65.0%), OR (temp > 31.0% AND humidity > 75.0%)
df_wheat["spoilage_label"] = (
    (df_wheat["moisture_content"] > 13.5) |
    ((df_wheat["moisture_content"] > 12.5) & (df_wheat["warehouse_temp"] > 28.0) & (df_wheat["humidity"] > 65.0)) |
    ((df_wheat["warehouse_temp"] > 31.0) & (df_wheat["humidity"] > 75.0))
).astype(int)

# ── 3. SAVE CLEANED FILES ────────────────────────────────────────────────────
final_cols = [
    "commodity_encoded", "tonnage", "market_arrivals", "current_price",
    "rainfall_deficit", "warehouse_temp", "humidity", "moisture_content",
    "future_price", "spoilage_label"
]

df_rice[final_cols].to_csv(os.path.join(BASE_DIR, "cleaned_rice_data.csv"), index=False)
df_wheat[final_cols].to_csv(os.path.join(BASE_DIR, "cleaned_wheat_data.csv"), index=False)

print(f"SUCCESS: Cleaned Rice Data ({len(df_rice)} rows) saved.")
print(f"SUCCESS: Cleaned Wheat Data ({len(df_wheat)} rows) saved.")
print(f"Rice High Risk count: {df_rice['spoilage_label'].sum()} / {len(df_rice)}")
print(f"Wheat High Risk count: {df_wheat['spoilage_label'].sum()} / {len(df_wheat)}")
