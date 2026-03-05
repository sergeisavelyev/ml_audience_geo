from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h3
import osmnx as ox


@dataclass(frozen=True)
class OSMConfig:
    h3_resolution: int = 9
    buffer_deg: float = 0.08  # ~5-10 км вокруг Казани (приблизительно)

    poi_tags: Dict[str, bool] = None

    highway_keep: Tuple[str, ...] = (
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "service"
    )

    landuse_keep: Tuple[str, ...] = (
        "residential", "commercial", "retail", "industrial",
        "forest", "grass", "recreation_ground", "cemetery"
    )

    top_poi_types: int = 80

    def __post_init__(self):
        if self.poi_tags is None:
            object.__setattr__(self, "poi_tags", {
                "amenity": True,
                "shop": True,
                "office": True,
                "leisure": True,
                "tourism": True,
            })


class OSMFeatureExtractor:
    def __init__(self, config: Optional[OSMConfig] = None):
        self.config = config or OSMConfig()

    def build_features(self, targets_df: pd.DataFrame) -> pd.DataFrame:
        """
        targets_df: таблица с колонкой h3_9
        returns: DataFrame h3_9 + feature columns
        """
        hex_ids = targets_df["h3_9"].dropna().unique().tolist()
        min_lat, min_lon, max_lat, max_lon = self._bbox_from_hexes(hex_ids)


        bbox = (min_lon, min_lat, max_lon, max_lat)  # (west, south, east, north)

        poi = ox.features_from_bbox(
            bbox=bbox,
            tags=self.config.poi_tags
        )     


        roads = ox.features_from_bbox(
            bbox=bbox,
            tags={"highway": True}
        )      


        landuse = ox.features_from_bbox(
            bbox=bbox,
            tags={"landuse": True}
        )        

        buildings = ox.features_from_bbox(
            bbox=bbox,
            tags={"building": True}
        )   

        poi_features = self._aggregate_poi(poi)
        road_features = self._aggregate_roads(roads)
        landuse_features = self._aggregate_landuse(landuse)
        building_features = self._aggregate_buildings(buildings)        

        base = pd.DataFrame({"h3_9": hex_ids})

        df = (
            base.merge(poi_features, on="h3_9", how="left")
                .merge(road_features, on="h3_9", how="left")
                .merge(landuse_features, on="h3_9", how="left")
                .merge(building_features, on="h3_9", how="left")
        )
        df = df.copy() 

        # fill missing -> 0 for numeric counts/lengths
        for c in df.columns:
            if c != "h3_9":
                df[c] = df[c].fillna(0)


        # latlon = df["h3_9"].apply(h3.cell_to_latlng).tolist()
        latlon = pd.DataFrame(
            df["h3_9"].apply(h3.cell_to_latlng).tolist(),
            columns=["lat_c", "lon_c"],
            index=df.index,
        )

        df = pd.concat([df, latlon], axis=1)
        df = df.copy()        
        df[["lat_c", "lon_c"]] = pd.DataFrame(latlon, index=df.index)     

        return df

    def _bbox_from_hexes(self, hex_ids: List[str]) -> Tuple[float, float, float, float]:
        lats, lons = [], []
        for hx in hex_ids:
            lat, lon = h3.cell_to_latlng(hx)
            lats.append(lat)
            lons.append(lon)

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        b = self.config.buffer_deg
        return (min_lat - b, min_lon - b, max_lat + b, max_lon + b)

    def _hex_id(self, lat: float, lon: float) -> str:
        return h3.latlng_to_cell(lat, lon, self.config.h3_resolution)

    def _aggregate_poi(self, poi_gdf) -> pd.DataFrame:
        if poi_gdf is None or len(poi_gdf) == 0:
            return pd.DataFrame(columns=["h3_9"])

        g = poi_gdf.copy()

        # centroid for any geometry type
        g = g.to_crs(epsg=3857)
        g["centroid"] = g.geometry.centroid
        # если нужен length:
        # g["geom_len"] = g.geometry.length
        g = g.to_crs(epsg=4326)
        g["lat"] = g["centroid"].y
        g["lon"] = g["centroid"].x
        g["h3_9"] = g.apply(lambda r: self._hex_id(r["lat"], r["lon"]), axis=1)

        def pick_type(row) -> str:
            for k in ["amenity", "shop", "office", "leisure", "tourism"]:
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    return f"{k}:{v}"
            return "other"

        g["poi_type"] = g.apply(pick_type, axis=1)

        top_types = g["poi_type"].value_counts().head(self.config.top_poi_types).index.tolist()
        g.loc[~g["poi_type"].isin(top_types), "poi_type"] = "other"

        agg = (
            g.groupby(["h3_9", "poi_type"])
             .size()
             .unstack(fill_value=0)
             .reset_index()
        )

        rename = {c: f"poi_{c.replace(':', '_')}" for c in agg.columns if c != "h3_9"}
        return agg.rename(columns=rename)

    def _aggregate_roads(self, roads_gdf) -> pd.DataFrame:
        if roads_gdf is None or len(roads_gdf) == 0:
            return pd.DataFrame(columns=["h3_9"])

        g = roads_gdf.copy()

        # 1) считаем геометрию в метрах (EPSG:3857)
        g_m = g.to_crs(epsg=3857)

        # centroid и длина в метрах
        centroid_m = g_m.geometry.centroid
        geom_len_m = g_m.geometry.length

        # 2) переносим centroid в lat/lon (EPSG:4326)
        centroid_ll = centroid_m.to_crs(epsg=4326)

        g["lat"] = centroid_ll.y.values
        g["lon"] = centroid_ll.x.values
        g["geom_len"] = geom_len_m.values

        # 3) hex id
        g["h3_9"] = g.apply(lambda r: self._hex_id(r["lat"], r["lon"]), axis=1)

        def normalize_highway(v) -> str:
            if isinstance(v, list) and v:
                v = v[0]
            if not isinstance(v, str):
                return "other"
            return v if v in self.config.highway_keep else "other"

        g["highway_norm"] = g["highway"].apply(normalize_highway)

        # counts по типу дорог
        cnt = (
            g.groupby(["h3_9", "highway_norm"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        cnt = cnt.rename(columns={c: f"road_cnt_{c}" for c in cnt.columns if c != "h3_9"})

        # суммарная длина дорог в метрах по типам
        ln = (
            g.groupby(["h3_9", "highway_norm"])["geom_len"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        ln = ln.rename(columns={c: f"road_len_{c}" for c in ln.columns if c != "h3_9"})

        return cnt.merge(ln, on="h3_9", how="left")

    def _aggregate_landuse(self, landuse_gdf) -> pd.DataFrame:
        if landuse_gdf is None or len(landuse_gdf) == 0:
            return pd.DataFrame(columns=["h3_9"])

        g = landuse_gdf.copy()

        g = g.to_crs(epsg=3857)
        g["centroid"] = g.geometry.centroid
        # если нужен length:
        # g["geom_len"] = g.geometry.length
        g = g.to_crs(epsg=4326)
        g["lat"] = g["centroid"].y
        g["lon"] = g["centroid"].x
        g["h3_9"] = g.apply(lambda r: self._hex_id(r["lat"], r["lon"]), axis=1)

        def normalize_landuse(v) -> str:
            if not isinstance(v, str):
                return "other"
            return v if v in self.config.landuse_keep else "other"

        g["landuse_norm"] = g["landuse"].apply(normalize_landuse)

        agg = (
            g.groupby(["h3_9", "landuse_norm"])
             .size()
             .unstack(fill_value=0)
             .reset_index()
        )

        rename = {c: f"landuse_cnt_{c}" for c in agg.columns if c != "h3_9"}
        return agg.rename(columns=rename)
    
    def _aggregate_buildings(self, buildings_gdf) -> pd.DataFrame:

        if buildings_gdf is None or len(buildings_gdf) == 0:
            return pd.DataFrame(columns=["h3_9"])

        g = buildings_gdf.copy()

        # работаем в метрах
        g = g.to_crs(epsg=3857)

        # площадь здания
        g["area"] = g.geometry.area

        # centroid
        g["centroid"] = g.geometry.centroid

        # назад в lat/lon
        g = g.to_crs(epsg=4326)

        g["lat"] = g["centroid"].y
        g["lon"] = g["centroid"].x

        g["h3_9"] = g.apply(
            lambda r: self._hex_id(r["lat"], r["lon"]),
            axis=1
        )

        def normalize_building(v):

            if not isinstance(v, str):
                return "other"

            if v in [
                "residential",
                "apartments",
                "house"
            ]:
                return "residential"

            if v in [
                "commercial",
                "retail",
                "office"
            ]:
                return "commercial"

            if v in [
                "industrial",
                "warehouse"
            ]:
                return "industrial"

            return "other"

        g["building_type"] = g["building"].apply(normalize_building)

        # количество зданий
        cnt = (
            g.groupby(["h3_9", "building_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        cnt = cnt.rename(
            columns={
                c: f"building_cnt_{c}"
                for c in cnt.columns if c != "h3_9"
            }
        )

        # площадь зданий
        area = (
            g.groupby(["h3_9", "building_type"])["area"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )

        area = area.rename(
            columns={
                c: f"building_area_{c}"
                for c in area.columns if c != "h3_9"
            }
        )

        return cnt.merge(area, on="h3_9", how="left")