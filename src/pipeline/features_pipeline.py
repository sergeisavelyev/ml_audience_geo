from __future__ import annotations

import pandas as pd

from src.features.osm_feature_extractor import OSMFeatureExtractor, OSMConfig


def main() -> None:
    targets = pd.read_parquet("data/kazan_targets_h3.parquet")

    extractor = OSMFeatureExtractor(OSMConfig(h3_resolution=9, buffer_deg=0.08))
    features = extractor.build_features(targets)

    print("features.shape =", features.shape)
    print("poi cols =", sum(c.startswith("poi_") for c in features.columns))
    print("road cols =", sum(c.startswith("road_") for c in features.columns))
    print("landuse cols =", sum(c.startswith("landuse_") for c in features.columns))    

    print("features.shape =", features.shape)
    print("first cols:", list(features.columns[:8]))
    print("last cols:", list(features.columns[-8:]))

    print(features.filter(like="road_len_").describe().T.head(10))

    features.to_parquet("data/kazan_hex_features_osm.parquet", index=False)
    print("Saved: data/kazan_hex_features_osm.parquet")


if __name__ == "__main__":
    main()