import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/emergency_calls.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("hour")["calls"].mean().plot(ax=axes[0], title="Avg calls by hour")
df.groupby("zone_id")["calls"].mean().plot(kind="bar", ax=axes[1], title="Avg calls by zone")
df.groupby("is_rainy")["calls"].mean().plot(kind="bar", ax=axes[2], title="Rainy (1) vs dry (0)")
plt.tight_layout()
plt.savefig("data/overview.png")
plt.show()
