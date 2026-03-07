from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import h3
import osmnx as ox
import geopandas as gpd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class StudentDistanceConfig:
    h3_resolution: int = 9
    buffer_deg: float = 0.08


class StudentDistanceFeatureExtractor:
    def __init__(self, config: Optional[StudentDistanceConfig] = None):
        self.config = config or StudentDistanceConfig()

    def build_features(self, hex_df: pd.DataFrame) -> pd.DataFrame:
        hex_ids = hex_df["h3_9"].dropna().unique().tolist()
        centers = [h3.cell_to_latlng(hx) for hx in hex_ids]

        lat = np.array([c[0] for c in centers])
        lon = np.array([c[1] for c in centers])

        min_lat, max_lat = float(lat.min()), float(lat.max())
        min_lon, max_lon = float(lon.min()), float(lon.max())

        b = self.config.buffer_deg
        bbox = (min_lon - b, min_lat - b, max_lon + b, max_lat + b)

        g_centers = gpd.GeoDataFrame(
            {"h3_9": hex_ids},
            geometry=gpd.points_from_xy(lon, lat),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        university_pts = self._load_university_points(bbox)
        college_pts = self._load_college_points(bbox)
        dormitory_pts = self._load_dormitory_points(bbox)

        dist_to_university = self._min_distance_kdtree(g_centers, university_pts)
        dist_to_college = self._min_distance_kdtree(g_centers, college_pts)
        dist_to_dormitory = self._min_distance_kdtree(g_centers, dormitory_pts)

        out = pd.DataFrame(
            {
                "h3_9": hex_ids,
                "dist_to_university_m": dist_to_university,
                "dist_to_college_m": dist_to_college,
                "dist_to_dormitory_m": dist_to_dormitory,
            }
        )
        return out

    def _load_university_points(self, bbox) -> gpd.GeoDataFrame:
        g = ox.features_from_bbox(
            bbox=bbox,
            tags={"amenity": True},
        )
        if g is None or len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        g = g.reset_index()
        g = g[g["amenity"].astype(str).str.lower() == "university"]

        if len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        return self._to_centroid_points(g)

    def _load_college_points(self, bbox) -> gpd.GeoDataFrame:
        g = ox.features_from_bbox(
            bbox=bbox,
            tags={"amenity": True},
        )
        if g is None or len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        g = g.reset_index()
        g = g[g["amenity"].astype(str).str.lower() == "college"]

        if len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        return self._to_centroid_points(g)

    def _load_dormitory_points(self, bbox) -> gpd.GeoDataFrame:
        g = ox.features_from_bbox(
            bbox=bbox,
            tags={"building": True},
        )
        if g is None or len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        g = g.reset_index()
        g = g[g["building"].astype(str).str.lower() == "dormitory"]

        if len(g) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        return self._to_centroid_points(g)

    def _to_centroid_points(self, g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = gpd.GeoDataFrame(g, geometry=g.geometry, crs="EPSG:4326")
        g = g.to_crs(epsg=3857)
        g["centroid"] = g.geometry.centroid
        g = g.set_geometry("centroid").to_crs(epsg=4326)
        return g[["centroid"]].rename(columns={"centroid": "geometry"}).set_geometry("geometry")

    def _min_distance_kdtree(self, centers_3857: gpd.GeoDataFrame, points_4326: gpd.GeoDataFrame) -> np.ndarray:
        if points_4326 is None or len(points_4326) == 0:
            return np.full(len(centers_3857), 999999.0)

        pts = points_4326.to_crs(epsg=3857)
        xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
        tree = cKDTree(xy)

        q = np.column_stack([centers_3857.geometry.x.to_numpy(), centers_3857.geometry.y.to_numpy()])
        dist, _ = tree.query(q, k=1)
        return dist