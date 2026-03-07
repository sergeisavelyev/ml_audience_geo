from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import h3
import osmnx as ox
import geopandas as gpd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DistanceConfig:
    h3_resolution: int = 9
    buffer_deg: float = 0.08  # как в OSM extractor
    # центр Казани (можно поправить)
    city_center_lat: float = 55.796127
    city_center_lon: float = 49.106405


class DistanceFeatureExtractor:
    def __init__(self, config: Optional[DistanceConfig] = None):
        self.config = config or DistanceConfig()

    def build_features(self, hex_df: pd.DataFrame) -> pd.DataFrame:
        hex_ids = hex_df["h3_9"].dropna().unique().tolist()
        centers = [h3.cell_to_latlng(hx) for hx in hex_ids]  # [(lat, lon), ...]
        lat = np.array([c[0] for c in centers])
        lon = np.array([c[1] for c in centers])

        # bbox по центрам
        min_lat, max_lat = float(lat.min()), float(lat.max())
        min_lon, max_lon = float(lon.min()), float(lon.max())
        b = self.config.buffer_deg
        bbox = (min_lon - b, min_lat - b, max_lon + b, max_lat + b)  # (west,south,east,north)

        # превращаем центры в GeoDataFrame и переводим в метры
        g_centers = gpd.GeoDataFrame(
            {"h3_9": hex_ids},
            geometry=gpd.points_from_xy(lon, lat),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # 1) distance to center
        center_pt = gpd.GeoSeries(
            gpd.points_from_xy([self.config.city_center_lon], [self.config.city_center_lat]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857).iloc[0]
        dist_to_center = g_centers.geometry.distance(center_pt).to_numpy()

        # 2) метро (OSM)
        metro_pts = self._load_metro_points(bbox)
        dist_to_metro = self._min_distance_kdtree(g_centers, metro_pts)

        # 3) моллы/ТЦ (OSM)
        mall_pts = self._load_mall_points(bbox)
        dist_to_mall = self._min_distance_kdtree(g_centers, mall_pts)

        # 4) крупные дороги: берём линии major roads и считаем dist до линии
        major_roads = self._load_major_roads(bbox)
        dist_to_primary = self._min_distance_to_lines(g_centers, major_roads)

        out = pd.DataFrame({
            "h3_9": hex_ids,
            "dist_to_center_m": dist_to_center,
            "dist_to_metro_m": dist_to_metro,
            "dist_to_mall_m": dist_to_mall,
            "dist_to_primary_road_m": dist_to_primary,
        })
        return out

    def _load_metro_points(self, bbox) -> gpd.GeoDataFrame:
        # OSM по метро бывает разно размечен — берём “широко”
        tags = {
            "railway": True,
            "station": True,
            "public_transport": True,
            "subway": True,
        }
        g = ox.features_from_bbox(bbox=bbox, tags=tags)
        if g is None or len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # оставим только то, что похоже на метро/станции
        g = g.reset_index()
        cols = set(g.columns)

        def is_metro(row) -> bool:
            # эвристика: station=subway или subway=yes или railway=station + name
            v_station = str(row.get("station", "")).lower()
            v_subway = str(row.get("subway", "")).lower()
            v_pt = str(row.get("public_transport", "")).lower()
            v_rail = str(row.get("railway", "")).lower()
            if "subway" in v_station:
                return True
            if v_subway in ("yes", "true", "1"):
                return True
            if v_pt in ("station", "stop_position", "platform") and v_rail in ("station", "subway_entrance"):
                return True
            if v_rail in ("subway_entrance",):
                return True
            return False

        g = g[g.apply(is_metro, axis=1)]
        if len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # точки: берем centroid
        g = gpd.GeoDataFrame(g, geometry=g.geometry, crs="EPSG:4326")
        g = g.to_crs(epsg=3857)
        g["centroid"] = g.geometry.centroid
        g = g.set_geometry("centroid").to_crs(epsg=4326)
        return g[["centroid"]].rename(columns={"centroid": "geometry"}).set_geometry("geometry")

    def _load_mall_points(self, bbox) -> gpd.GeoDataFrame:
        tags = {
            "shop": True,
            "amenity": True,
            "building": True,
        }
        g = ox.features_from_bbox(bbox=bbox, tags=tags)
        if g is None or len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        g = g.reset_index()

        def is_mall(row) -> bool:
            shop = str(row.get("shop", "")).lower()
            amen = str(row.get("amenity", "")).lower()
            bld = str(row.get("building", "")).lower()
            # эвристики: shop=mall или amenity=marketplace, крупные retail
            if shop == "mall":
                return True
            if amen in ("marketplace",):
                return True
            if bld in ("retail", "commercial") and shop in ("", "yes"):
                return True
            return False

        g = g[g.apply(is_mall, axis=1)]
        if len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        g = gpd.GeoDataFrame(g, geometry=g.geometry, crs="EPSG:4326")
        g = g.to_crs(epsg=3857)
        g["centroid"] = g.geometry.centroid
        g = g.set_geometry("centroid").to_crs(epsg=4326)
        return g[["centroid"]].rename(columns={"centroid": "geometry"}).set_geometry("geometry")

    def _load_major_roads(self, bbox) -> gpd.GeoDataFrame:
        g = ox.features_from_bbox(bbox=bbox, tags={"highway": True})
        if g is None or len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        g = g.reset_index()

        keep = {"motorway", "trunk", "primary"}
        def is_major(v) -> bool:
            if isinstance(v, list) and v:
                v = v[0]
            return isinstance(v, str) and v in keep

        g = g[g["highway"].apply(is_major)]
        if len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return gpd.GeoDataFrame(g, geometry=g.geometry, crs="EPSG:4326").to_crs(epsg=3857)

    def _min_distance_kdtree(self, centers_3857: gpd.GeoDataFrame, points_4326: gpd.GeoDataFrame) -> np.ndarray:
        # если точек нет — вернем большое расстояние
        if points_4326 is None or len(points_4326) == 0:
            return np.full(len(centers_3857), 999999.0)

        pts = points_4326.to_crs(epsg=3857)
        xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
        tree = cKDTree(xy)

        q = np.column_stack([centers_3857.geometry.x.to_numpy(), centers_3857.geometry.y.to_numpy()])
        dist, _ = tree.query(q, k=1)
        return dist

    def _min_distance_to_lines(self, centers_3857: gpd.GeoDataFrame, lines_3857: gpd.GeoDataFrame) -> np.ndarray:
        if lines_3857 is None or len(lines_3857) == 0:
            return np.full(len(centers_3857), 999999.0)

        # расстояние точки до набора линий “в лоб” может быть тяжелым,
        # но для Казани/1000 hex нормально. Потом оптимизируем.
        # unary_union делает один MultiLineString
        union = lines_3857.geometry.unary_union
        return centers_3857.geometry.distance(union).to_numpy()