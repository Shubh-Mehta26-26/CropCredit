import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 19.9975,
    "longitude": 73.7898,
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "daily": ["temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean"],
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame({
    "date": data["daily"]["time"],
    "warehouse_temp": data["daily"]["temperature_2m_mean"],
    "rainfall": data["daily"]["precipitation_sum"],
    "humidity": data["daily"]["relative_humidity_2m_mean"]
})

print(df.head())

df.to_csv("nashik_aug_2024_2025_weather.csv", index=False)

print("✅ Exact weather data ready!")