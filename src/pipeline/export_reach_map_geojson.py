from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3


IN_PATH = Path("data/kazan_reach_map_h3_9.parquet")
OUT_PATH = Path("data/kazan_reach_map_h3_9.geojson")


def h3_to_polygon(hx: str) -> Polygon:
    """
    Превращаем H3 cell -> shapely Polygon.
    В h3 границы обычно возвращаются как (lat, lon), нам нужно (lon, lat).
    """
    boundary = h3.cell_to_boundary(hx)  # list of (lat, lon)
    coords = [(lon, lat) for (lat, lon) in boundary]
    # замыкаем
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return Polygon(coords)


def main() -> None:
    df = pd.read_parquet(IN_PATH)

    # строим полигоны гексов
    geom = df["h3_9"].apply(h3_to_polygon)

    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    # GeoJSON удобен для Kepler/Folium/QGIS
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    print("Saved:", OUT_PATH)
    print("Rows:", len(gdf))
    print(gdf[["h3_9", "pred_totals"]].head())


if __name__ == "__main__":
    main()