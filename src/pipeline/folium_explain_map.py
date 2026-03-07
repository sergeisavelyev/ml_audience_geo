from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import folium
from branca.colormap import linear

from src.features.distance_feature_extractor import DistanceFeatureExtractor, DistanceConfig


MAP_GEOJSON = Path("data/kazan_reach_map_h3_9.geojson")
MAP_PARQUET = Path("data/kazan_reach_map_h3_9.parquet")
FEATURES_PARQUET = Path("data/kazan_hex_features_osm.parquet")
OUT_HTML = Path("data/kazan_explain_map.html")


def add_hex_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    layer_name: str,
    aliases: list[str] | None = None,
) -> None:
    aliases = aliases or [value_col]

    vmin = float(gdf[value_col].min())
    vmax = float(gdf[value_col].max())

    colormap = linear.YlOrRd_09.scale(vmin, vmax)

    def style_function(feature):
        value = feature["properties"].get(value_col, 0)

        if value is None or pd.isna(value):
            value = 0

        return {
            "fillColor": colormap(float(value)),
            "color": "black",
            "weight": 0.15,
            "fillOpacity": 0.65,
        }

    tooltip_fields = ["h3_9", value_col]
    tooltip_aliases = ["Hex", aliases[0]]

    fg = folium.FeatureGroup(name=layer_name, show=False)

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
        ),
    ).add_to(fg)

    fg.add_to(m)
    colormap.caption = layer_name
    colormap.add_to(m)


def add_point_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    color: str = "blue",
    radius: int = 4,
) -> None:
    fg = folium.FeatureGroup(name=layer_name, show=False)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lat = geom.y
        lon = geom.x

        popup_text = row.get("name", layer_name)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=str(popup_text),
        ).add_to(fg)

    fg.add_to(m)


def main() -> None:
    # 1. Читаем карту и признаки
    map_gdf = gpd.read_file(MAP_GEOJSON)
    map_df = pd.read_parquet(MAP_PARQUET)
    feats_df = pd.read_parquet(FEATURES_PARQUET)

    # 2. Объединяем всё в один gdf
    gdf = map_gdf.merge(
        feats_df[
            [
                "h3_9",
                "dist_to_center_m",
                "dist_to_metro_m",
                "dist_to_mall_m",
                "dist_to_primary_road_m",
                "building_area_residential",
            ]
        ],
        on="h3_9",
        how="left",
    )

    feature_cols = [
        "pred_totals",
        "dist_to_center_m",
        "dist_to_metro_m",
        "dist_to_mall_m",
        "dist_to_primary_road_m",
        "building_area_residential",
    ]

    gdf[feature_cols] = gdf[feature_cols].fillna(0)    

    # 3. Создаём карту
    center_lat = float(map_df["lat_c"].mean())
    center_lon = float(map_df["lon_c"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    # 4. Основной слой reach
    add_hex_layer(
        m,
        gdf,
        value_col="pred_totals",
        layer_name="Predicted Reach",
        aliases=["Reach"],
    )

    # 5. Объясняющие слои
    add_hex_layer(
        m,
        gdf,
        value_col="dist_to_metro_m",
        layer_name="Distance to Metro (m)",
        aliases=["Distance to Metro"],
    )

    add_hex_layer(
        m,
        gdf,
        value_col="dist_to_mall_m",
        layer_name="Distance to Mall (m)",
        aliases=["Distance to Mall"],
    )

    add_hex_layer(
        m,
        gdf,
        value_col="dist_to_center_m",
        layer_name="Distance to Center (m)",
        aliases=["Distance to Center"],
    )

    add_hex_layer(
        m,
        gdf,
        value_col="building_area_residential",
        layer_name="Residential Building Area",
        aliases=["Residential Building Area"],
    )

    # Покажем reach по умолчанию
    # Способ простой: первый слой уже добавлен, а остальные можно оставить скрытыми.

    # 6. Подгружаем точки метро и моллов
    dist_extractor = DistanceFeatureExtractor(DistanceConfig())
    hex_ids = feats_df["h3_9"].dropna().unique().tolist()

    # bbox по центрам hex
    centers = [tuple(x) for x in feats_df[["lon_c", "lat_c"]].drop_duplicates().to_numpy()]
    min_lon = min(x[0] for x in centers)
    max_lon = max(x[0] for x in centers)
    min_lat = min(x[1] for x in centers)
    max_lat = max(x[1] for x in centers)
    b = dist_extractor.config.buffer_deg
    bbox = (min_lon - b, min_lat - b, max_lon + b, max_lat + b)

    metro = dist_extractor._load_metro_points(bbox)
    malls = dist_extractor._load_mall_points(bbox)

    if len(metro) > 0:
        add_point_layer(m, metro, layer_name="Metro", color="blue", radius=4)

    if len(malls) > 0:
        add_point_layer(m, malls, layer_name="Malls", color="green", radius=4)

    # 7. Переключатель слоёв
    folium.LayerControl(collapsed=False).add_to(m)

    print(gdf[feature_cols].isna().sum())       

    # 8. Сохраняем
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"Saved explain map → {OUT_HTML}")


if __name__ == "__main__":
    main()