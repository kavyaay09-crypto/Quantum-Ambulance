import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Base emergency rate per hour for each of the 8 zones
base_rate = np.array([1.2, 0.5, 0.8, 0.7, 0.6, 0.5, 1.5, 0.4])

# Hour-of-day multiplier: busier in morning rush and evening
hour_mult = np.array([
    0.5, 0.4, 0.3, 0.3, 0.4, 0.6, 0.9, 1.3, 1.5, 1.1, 1.0, 1.0,
    1.1, 1.0, 1.0, 1.1, 1.3, 1.6, 1.8, 1.7, 1.4, 1.1, 0.8, 0.6
])

rows = []
start = pd.Timestamp("2026-07-01")
for day in range(90):                       # 90 days of history
    rainy = rng.random() < 0.25             # 25% of days are rainy
    event_zone = rng.integers(0, 8) if rng.random() < 0.1 else -1  # random local event
    for hour in range(24):
        ts = start + pd.Timedelta(days=day, hours=hour)
        weekend = int(ts.dayofweek >= 5)
        for z in range(8):
            rate = base_rate[z] * hour_mult[hour]
            if rainy:
                rate *= 1.4
            if weekend and z in (0, 4):     # downtown and old town busier on weekends
                rate *= 1.3
            if z == event_zone:
                rate *= 2.5
            calls = rng.poisson(rate)
            rows.append([ts, z, hour, ts.dayofweek, weekend, int(rainy), calls])

df = pd.DataFrame(rows, columns=["timestamp", "zone_id", "hour", "day_of_week",
                                 "is_weekend", "is_rainy", "calls"])
df.to_csv("data/emergency_calls.csv", index=False)
print(df.head(10))
print("\nTotal rows:", len(df))
print("\nAverage calls per zone per hour:")
print(df.groupby("zone_id")["calls"].mean().round(2))
