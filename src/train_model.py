import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/emergency_calls.csv", parse_dates=["timestamp"])
df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

# Recent-history features, computed separately for each zone
g = df.groupby("zone_id")["calls"]
df["lag1"] = g.shift(1)
df["roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
df = df.dropna().reset_index(drop=True)

FEATURES = ["zone_id", "hour", "day_of_week", "is_weekend", "is_rainy", "lag1", "roll3"]

# Time-based split: train on the past, test on the last 14 days
split_time = df["timestamp"].max() - pd.Timedelta(days=14)
train = df[df["timestamp"] <= split_time].copy()
test = df[df["timestamp"] > split_time].copy()

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="count:poisson",
    random_state=42,
)
model.fit(train[FEATURES], train["calls"])
test["pred"] = model.predict(test[FEATURES])

# Simple baseline: average calls for that zone and hour in the training data
base = train.groupby(["zone_id", "hour"])["calls"].mean().rename("base")
test = test.join(base, on=["zone_id", "hour"])

mae_model = mean_absolute_error(test["calls"], test["pred"])
mae_base = mean_absolute_error(test["calls"], test["base"])
print(f"Baseline MAE (zone+hour average): {mae_base:.3f}")
print(f"XGBoost MAE:                      {mae_model:.3f}")

# Save the model and typical values per zone (used as defaults when predicting)
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "zone_means": train.groupby("zone_id")["calls"].mean().to_dict(),
    },
    "data/demand_model.joblib",
)

# Charts: predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
test.groupby("hour")[["calls", "pred"]].mean().plot(ax=axes[0], title="Avg calls by hour: actual vs predicted")
test.groupby("zone_id")[["calls", "pred"]].mean().plot(kind="bar", ax=axes[1], title="Avg calls by zone: actual vs predicted")
plt.tight_layout()
plt.savefig("data/model_eval.png")
plt.show()
