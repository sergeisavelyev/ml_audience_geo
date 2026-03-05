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


DATA_DIR = Path("data")
POLYGONS_PATH = DATA_DIR / "polygons.json"
MODEL_PATH = DATA_DIR / "models" / "reach_catboost_spatial_parent7.cbm"
OUT_PATH = DATA_DIR / "kazan_reach_map_h3_9.parquet"

H3_RES = 9


def _load_segment_polygons(path: Path) -> gpd.GeoDataFrame:
    """
    polygons.json: список сегментов, у каждого polygons[0].points = [{latitude, longitude}, ...]
    Возвращает GeoDataFrame с геометриями сегментов.
    """
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

        # shapely ждёт (lon, lat)
        coords = [(p["longitude"], p["latitude"]) for p in pts]
        try:
            geoms.append(Polygon(coords))
        except Exception:
            continue

    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
    return gdf


def _build_city_polygon(gdf: gpd.GeoDataFrame) -> Polygon | MultiPolygon:
    """
    Объединяем все сегменты в одну (или несколько) геометрию города.
    """
    union_geom = unary_union(gdf.geometry.values)
    return union_geom


def _polyfill_h3(city_geom: Polygon | MultiPolygon, res: int) -> List[str]:
    cells: set[str] = set()

    def exterior_lonlat(poly: Polygon) -> List[Tuple[float, float]]:
        # shapely coords already (lon, lat)
        return list(poly.exterior.coords)

    if isinstance(city_geom, Polygon):
        ring = exterior_lonlat(city_geom)
        cells |= set(_h3_polygon_to_cells(ring, res))
    else:
        for poly in city_geom.geoms:
            ring = exterior_lonlat(poly)
            cells |= set(_h3_polygon_to_cells(ring, res))

    return sorted(cells)


def _h3_polygon_to_cells(lonlat_ring: List[Tuple[float, float]], res: int) -> List[str]:
    """
    В вашей версии h3.polygon_to_cells ждёт H3Shape (LatLngPoly),
    а не GeoJSON dict.
    lonlat_ring: список (lon, lat), замкнутый (последняя = первая) или нет.
    """
    # h3 LatLngPoly ждёт координаты в формате (lat, lon)
    latlng = [(lat, lon) for (lon, lat) in lonlat_ring]

    # на всякий случай замкнём кольцо
    if latlng[0] != latlng[-1]:
        latlng.append(latlng[0])

    poly = h3.LatLngPoly(latlng)
    return list(h3.polygon_to_cells(poly, res))


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

    # небольшой буфер, чтобы “докрыть” дырки и воду (в метрах, потом обратно)
    # Можно отключить, если хочешь строго по сегментам.
    city_geom_m = gpd.GeoSeries([city_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    city_geom_m = city_geom_m.buffer(250)  # 250м
    city_geom = gpd.GeoSeries([city_geom_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]

    print("3) H3 polyfill at res:", H3_RES)
    hexes = _polyfill_h3(city_geom, H3_RES)
    print("Hexes in city:", len(hexes))

    # 4) Feature extraction for ALL hexes
    print("4) Build OSM features (may take time on first run)")
    extractor = OSMFeatureExtractor(OSMConfig(h3_resolution=H3_RES, buffer_deg=0.05))
    hex_df = pd.DataFrame({"h3_9": hexes})
    feats = extractor.build_features(hex_df)
    print("Features shape:", feats.shape)

    # 5) Predict reach using trained model
    print("5) Load model:", MODEL_PATH)
    model = _load_model(MODEL_PATH)

    X = feats.drop(columns=["h3_9"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0)
    pred_log = model.predict(X)
    pred_totals = np.expm1(pred_log)

    out = feats[["h3_9", "lat_c", "lon_c"]].copy()
    out["pred_totals"] = pred_totals

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print("Saved map to:", OUT_PATH)
    print(out.head())


if __name__ == "__main__":
    main()