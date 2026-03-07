from __future__ import annotations

import pandas as pd

from src.features.osm_feature_extractor import OSMFeatureExtractor, OSMConfig
from src.features.distance_feature_extractor import DistanceFeatureExtractor
from src.features.spatial_lag_feature_extractor import (
    SpatialLagFeatureExtractor,
    SpatialLagConfig,
)
from src.features.student_distance_feature_extractor import (
    StudentDistanceFeatureExtractor,
    StudentDistanceConfig,
)


def main() -> None:

    # 1 загрузка таргетов
    targets = pd.read_parquet("data/kazan_targets_h3.parquet")

    # 2 OSM признаки
    osm_extractor = OSMFeatureExtractor(OSMConfig(h3_resolution=9, buffer_deg=0.08))
    osm_features = osm_extractor.build_features(targets)

    print("OSM features shape =", osm_features.shape)

    print("poi cols =", sum(c.startswith("poi_") for c in osm_features.columns))
    print("road cols =", sum(c.startswith("road_") for c in osm_features.columns))
    print("landuse cols =", sum(c.startswith("landuse_") for c in osm_features.columns))
    print("building cols =", sum("building" in c for c in osm_features.columns))

    # 3 distance признаки
    dist_extractor = DistanceFeatureExtractor()
    dist_features = dist_extractor.build_features(osm_features[["h3_9"]])

    print("distance features shape =", dist_features.shape)
    print(dist_features.head())

    student_extractor = StudentDistanceFeatureExtractor(
        StudentDistanceConfig(h3_resolution=9, buffer_deg=0.08)
    )
    student_features = student_extractor.build_features(targets[["h3_9"]])

    print("student distance features shape =", student_features.shape)
    print(student_features.head())    

    # 4 объединяем всё
    features = (
        osm_features
        .merge(dist_features, on="h3_9", how="left")
        .merge(student_features, on="h3_9", how="left")
    )
    # -------------------------------
    # Spatial lag features
    # -------------------------------

    lag_cols = [
        "road_len_service",
        "road_len_other",
        "road_len_residential",
        "building_area_residential",
        "dist_to_center_m",
        "dist_to_metro_m",
        "dist_to_mall_m",
    ]

    lag_extractor = SpatialLagFeatureExtractor(
        SpatialLagConfig(
            h3_col="h3_9",
            feature_cols=lag_cols,
        )
    )

    lag_features = lag_extractor.build_features(features)

    print("lag features shape =", lag_features.shape)

    features = features.merge(lag_features, on="h3_9", how="left")    

    print("FINAL features shape =", features.shape)

    print("first cols:", list(features.columns[:8]))
    print("last cols:", list(features.columns[-8:]))

    print(features.filter(like="road_len_").describe().T.head(10))

    # 5 сохраняем
    features.to_parquet("data/kazan_hex_features_osm.parquet", index=False)

    print("Saved: data/kazan_hex_features_osm.parquet")


if __name__ == "__main__":
    main()