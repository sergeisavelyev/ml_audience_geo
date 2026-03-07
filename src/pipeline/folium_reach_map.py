from __future__ import annotations

import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import linear


def main():

    gdf = gpd.read_file("data/kazan_reach_map_h3_9.geojson")

    center_lat = gdf["lat_c"].mean()
    center_lon = gdf["lon_c"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    colormap = linear.YlOrRd_09.scale(
        gdf["pred_totals"].min(),
        gdf["pred_totals"].max()
    )

    def style_function(feature):

        reach = feature["properties"]["pred_totals"]

        return {
            "fillColor": colormap(reach),
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.7
        }

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["pred_totals", "h3_9"],
            aliases=["Reach", "Hex"]
        )
    ).add_to(m)

    colormap.caption = "Predicted Reach"
    colormap.add_to(m)

    m.save("data/kazan_reach_map_folium.html")

    print("Saved map → data/kazan_reach_map_folium.html")


if __name__ == "__main__":
    main()