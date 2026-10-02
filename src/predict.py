import joblib
import pandas as pd

_art = joblib.load("data/demand_model.joblib")
NUM_ZONES = 8

def predict_demand(hour, day_of_week, is_rainy, lag1=None, roll3=None):
    """Return expected emergency calls for each zone in the next hour.

    lag1 / roll3: optional lists of 8 numbers (recent calls per zone).
    If not given, typical values for each zone are used.
    """
    means = _art["zone_means"]
    rows = []
    for z in range(NUM_ZONES):
        l1 = lag1[z] if lag1 is not None else means[z]
        r3 = roll3[z] if roll3 is not None else means[z]
        rows.append([z, hour, day_of_week, int(day_of_week >= 5), int(is_rainy), l1, r3])
    X = pd.DataFrame(rows, columns=_art["features"])
    preds = _art["model"].predict(X)
    return pd.DataFrame({"zone_id": range(NUM_ZONES), "expected_calls": preds.round(2)})

if __name__ == "__main__":
    print("Normal weekday, 6 PM, dry:")
    print(predict_demand(hour=18, day_of_week=2, is_rainy=False), "\n")

    print("Same time, rainy:")
    print(predict_demand(hour=18, day_of_week=2, is_rainy=True), "\n")

    print("Surge: sudden spike of calls in zone 6 (Highway Junction):")
    spike = [1, 1, 1, 1, 1, 1, 8, 1]
    print(predict_demand(hour=18, day_of_week=2, is_rainy=True, lag1=spike, roll3=spike))
