import requests
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# -----------------------------
# 1. GET AIR QUALITY DATA
# -----------------------------

latitude = 13.0827
longitude = 80.2707

url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone",
    "past_days": 30,
    "forecast_days": 1,
    "timezone": "auto"
}

response = requests.get(url, params=params)

if response.status_code != 200:
    print("API request failed.")
    print("Status code:", response.status_code)
    exit()


data = response.json()

df = pd.DataFrame(data["hourly"])

df["time"] = pd.to_datetime(df["time"])


# -----------------------------
# 2. FEATURE ENGINEERING
# -----------------------------

df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek

# Previous hour values
df["pm2_5_lag1"] = df["pm2_5"].shift(1)
df["pm10_lag1"] = df["pm10"].shift(1)
df["no2_lag1"] = df["nitrogen_dioxide"].shift(1)
df["so2_lag1"] = df["sulphur_dioxide"].shift(1)
df["ozone_lag1"] = df["ozone"].shift(1)

# Target = next hour PM2.5
df["target_pm2_5"] = df["pm2_5"].shift(-1)

df = df.dropna()


# -----------------------------
# 3. SELECT FEATURES & TARGET
# -----------------------------

features = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "day_of_week",
    "pm2_5_lag1",
    "pm10_lag1",
    "no2_lag1",
    "so2_lag1",
    "ozone_lag1"
]

X = df[features]
y = df["target_pm2_5"]


# -----------------------------
# 4. TIME-BASED TRAIN/TEST SPLIT
# -----------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTRAINING SAMPLES:", len(X_train))
print("TESTING SAMPLES:", len(X_test))


# -----------------------------
# 5. TRAIN RANDOM FOREST
# -----------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)


# -----------------------------
# 6. MAKE PREDICTIONS
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# 7. EVALUATE MODEL
# -----------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


print("\nMODEL PERFORMANCE")
print("-------------------------")
print(f"MAE  : {mae:.2f} μg/m³")
print(f"RMSE : {rmse:.2f} μg/m³")
print(f"R²   : {r2:.3f}")


# -----------------------------
# 8. SHOW ACTUAL VS PREDICTED
# -----------------------------

results = pd.DataFrame({
    "time": df["time"].iloc[split_index:],
    "actual_pm2_5": y_test.values,
    "predicted_pm2_5": predictions
})

print("\nLAST 10 PREDICTIONS")
print(results.tail(10))