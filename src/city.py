import numpy as np
import pandas as pd

# 8 zones with (x, y) coordinates in km on a simple map
zones = pd.DataFrame({
    "zone_id": range(8),
    "name": ["Downtown", "Airport", "Hospital Area", "Industrial", "Old Town",
             "University", "Highway Junction", "Suburb"],
    "x": [5, 9, 6, 2, 4, 8, 7, 1],
    "y": [5, 8, 3, 7, 2, 1, 6, 1],
})

# Distance matrix (straight-line km between zone centers)
coords = zones[["x", "y"]].values
dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

NUM_AMBULANCES = 3

if __name__ == "__main__":
    zones.to_csv("data/zones.csv", index=False)
    pd.DataFrame(dist.round(2)).to_csv("data/distances.csv", index=False)
    print(zones)
    print("\nDistance matrix (km):")
    print(dist.round(1))
