from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import h3
from catboost import CatBoostRegressor

from src.features.osm_feature_extractor import OSMFeatureExtractor, OSMConfig
from src.features.distance_feature_extractor import DistanceFeatureExtractor, DistanceConfig
from src.features.spatial_lag_feature_extractor import (
    SpatialLagFeatureExtractor,
    SpatialLagConfig,
)
from src.features.student_distance_feature_extractor import (
    StudentDistanceFeatureExtractor,
    StudentDistanceConfig,
)


DATA_DIR = Path("data")
POLYGONS_PATH = DATA_DIR / "polygons.json"

# Какой таргет предсказываем:
TARGET_COL = "age_18_24"
MODEL_PATH = DATA_DIR / "models" / f"{TARGET_COL}_catboost_spatial_parent7.cbm"
OUT_PATH = DATA_DIR / f"kazan_{TARGET_COL}_map_h3_9.parquet"

H3_RES = 9


def _load_segment_polygons(path: Path) -> gpd.GeoDataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    geoms = []
    for seg in data:
        polys = seg.get("polygons") or []
        if not polys:
            continue

        pts = polys[0].get("points") or []
        if len(pts) < 3:
            continue

        coords = [(p["longitude"], p["latitude"]) for p in pts]
        try:
            geoms.append(Polygon(coords))
        except Exception:
            continue

    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def _build_city_polygon(gdf: gpd.GeoDataFrame) -> Polygon | MultiPolygon:
    return unary_union(gdf.geometry.values)


def _h3_polygon_to_cells(lonlat_ring: List[Tuple[float, float]], res: int) -> List[str]:
    latlng = [(lat, lon) for (lon, lat) in lonlat_ring]
    if latlng[0] != latlng[-1]:
        latlng.append(latlng[0])

    poly = h3.LatLngPoly(latlng)
    return list(h3.polygon_to_cells(poly, res))


def _polyfill_h3(city_geom: Polygon | MultiPolygon, res: int) -> List[str]:
    cells: set[str] = set()

    def exterior_lonlat(poly: Polygon) -> List[Tuple[float, float]]:
        return list(poly.exterior.coords)

    if isinstance(city_geom, Polygon):
        cells |= set(_h3_polygon_to_cells(exterior_lonlat(city_geom), res))
    else:
        for poly in city_geom.geoms:
            cells |= set(_h3_polygon_to_cells(exterior_lonlat(poly), res))

    return sorted(cells)


def _load_model(path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(path)
    return model


def main() -> None:
    print("1) Load polygons:", POLYGONS_PATH)
    seg_gdf = _load_segment_polygons(POLYGONS_PATH)
    print("Segments polygons:", len(seg_gdf))

    print("2) Union polygons -> city geometry")
    city_geom = _build_city_polygon(seg_gdf)

    # небольшой буфер для более целостной городской формы
    city_geom_m = gpd.GeoSeries([city_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    city_geom_m = city_geom_m.buffer(250)
    city_geom = gpd.GeoSeries([city_geom_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]

    print("3) H3 polyfill at res:", H3_RES)
    hexes = _polyfill_h3(city_geom, H3_RES)
    print("Hexes in city:", len(hexes))

    hex_df = pd.DataFrame({"h3_9": hexes})

    # 4) OSM features
    print("4) Build OSM features")
    osm_extractor = OSMFeatureExtractor(OSMConfig(h3_resolution=H3_RES, buffer_deg=0.05))
    osm_features = osm_extractor.build_features(hex_df)

    # 5) Distance features
    print("5) Build distance features")
    dist_extractor = DistanceFeatureExtractor(DistanceConfig(h3_resolution=H3_RES, buffer_deg=0.05))
    dist_features = dist_extractor.build_features(hex_df)

    # 5.1) Student distance features
    print("5.1) Build student distance features")
    student_extractor = StudentDistanceFeatureExtractor(
        StudentDistanceConfig(h3_resolution=H3_RES, buffer_deg=0.05)
    )
    student_features = student_extractor.build_features(hex_df)
    print("Student features shape:", student_features.shape)

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
    print("Lag features shape:", lag_features.shape)

    features = features.merge(lag_features, on="h3_9", how="left")

    print("Features shape:", features.shape)

    # 6) Load model
    print("6) Load model:", MODEL_PATH)
    model = _load_model(MODEL_PATH)

    X = features.drop(columns=["h3_9"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)

    pred = model.predict(X)
    pred = np.clip(pred, 0.0, 1.0)

    out = features[["h3_9", "lat_c", "lon_c"]].copy()
    out[f"pred_{TARGET_COL}"] = pred

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("Saved map to:", OUT_PATH)
    print(out.head())
    print(out[f"pred_{TARGET_COL}"].describe())


if __name__ == "__main__":
    main()