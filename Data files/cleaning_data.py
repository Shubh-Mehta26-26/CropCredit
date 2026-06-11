

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==============================
# STEP 1: LOAD DATA
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Tomato.csv"), skiprows=3)


print("Original Shape:", df.shape)
print(df.head())


# ==============================
# STEP 2: CLEAN COLUMN NAMES
# ==============================
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "Arrival Date": "date",
    "Arrivals (Metric Tonnes)": "market_arrivals",
    "Modal Price (Rs./Quintal)": "current_price",
    "Minimum Price (Rs./Quintal)": "min_price",
    "Maximum Price (Rs./Quintal)": "max_price",
    "Variety": "commodity"
})


# ==============================
# STEP 3: DATA CLEANING
# ==============================
# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Convert numeric columns
cols = ["market_arrivals", "current_price", "min_price", "max_price"]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing values
df = df.dropna()

# Remove invalid values
df = df[(df["market_arrivals"] > 0) & (df["current_price"] > 0)]


# ==============================
# STEP 4: SET COMMODITY = ONION
# ==============================
df["commodity"] = "Tomato"


# ==============================
# STEP 5: ADD HYBRID FEATURES
# ==============================
# Onion-specific realistic ranges

df["tonnage"] = np.random.uniform(200, 2000, len(df))  # onion storage bulk
df["moisture_content"] = np.random.uniform(12, 20, len(df))  # onion moisture higher

df["warehouse_temp"] = np.random.uniform(18, 32, len(df))  # onion storage temp
df["humidity"] = np.random.uniform(60, 85, len(df))  # onions sensitive to humidity
df["rainfall_deficit"] = np.random.normal(15, 25, len(df))


# ==============================
# STEP 6: ENCODE COMMODITY
# ==============================
le = LabelEncoder()
df["commodity_encoded"] = le.fit_transform(df["commodity"])


# ==============================
# STEP 7: CREATE TARGETS
# ==============================
# Price prediction target
df["future_price"] = df["current_price"] * (
    1 + 0.08 * np.random.randn(len(df))  # onion has high volatility
)

# Spoilage risk (onion-specific)
df["spoilage_score"] = (
    0.25 * (df["warehouse_temp"] / 35) +
    0.35 * (df["humidity"] / 100) +
    0.40 * (df["moisture_content"] / 25)
)

df["spoilage_label"] = (df["spoilage_score"] > 0.5).astype(int)


# ==============================
# STEP 8: FINAL DATASET
# ==============================
final_columns = [
    "commodity_encoded",
    "tonnage",
    "market_arrivals",
    "current_price",
    "rainfall_deficit",
    "warehouse_temp",
    "humidity",
    "moisture_content",
    "future_price",
    "spoilage_label"
]

df_final = df[final_columns]

print("\nFinal Cleaned Wheat Dataset:")
print(df_final.head())

print("\nShape:", df_final.shape)


# ==============================
# STEP 9: SAVE FILE
# ==============================
df_final.to_csv(os.path.join(BASE_DIR, "cleaned_Tomato_data.csv"), index=False)

print("\n[SUCCESS] Dataset cleaned and saved successfully!")