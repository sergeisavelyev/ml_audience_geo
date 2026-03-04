import json
import pandas as pd
import numpy as np


class PolygonLoader:

    def __init__(self, path):
        self.path = path

    def load(self):

        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def to_centroid_dataframe(self):

        data = self.load()

        rows = []

        for seg in data:

            segment_id = seg["id"]

            points = seg["polygons"][0]["points"]

            lats = [p["latitude"] for p in points]
            lons = [p["longitude"] for p in points]

            lat = np.mean(lats)
            lon = np.mean(lons)

            rows.append({
                "segment_id": segment_id,
                "lat": lat,
                "lon": lon
            })

        return pd.DataFrame(rows)