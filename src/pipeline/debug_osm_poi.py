import osmnx as ox
import pandas as pd

bbox = (49.0, 55.7, 49.3, 55.9)  # примерно Казань

poi = ox.features_from_bbox(
    bbox=bbox,
    tags={
        "amenity": True,
        "shop": True,
        "office": True,
        "leisure": True,
        "tourism": True,
    }
)

print("Total objects:", len(poi))

cols = ["amenity", "shop", "office", "leisure", "tourism"]

for c in cols:
    if c in poi.columns:
        print("\nTop", c)
        print(poi[c].value_counts().head(15))