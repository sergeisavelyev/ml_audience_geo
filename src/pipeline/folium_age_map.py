from pathlib import Path

import pandas as pd
import folium
from h3 import cell_to_boundary
from branca.colormap import linear


DATA_PATH = Path("data/kazan_age_18_24_map_h3_9.parquet")
OUT_HTML = Path("data/kazan_age_18_24_map.html")


def hex_to_geojson(hex_id, value):

    boundary = cell_to_boundary(hex_id)

    # folium ожидает [lon, lat]
    coords = [[lon, lat] for lat, lon in boundary]

    # закрываем полигон
    coords.append(coords[0])

    return {
        "type": "Feature",
        "properties": {
            "value": value
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords]
        },
    }


def main():

    df = pd.read_parquet(DATA_PATH)

    print(df["pred_age_18_24"].describe())

    center_lat = df["lat_c"].mean()
    center_lon = df["lon_c"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    colormap = linear.YlOrRd_09.scale(
        df["pred_age_18_24"].min(),
        df["pred_age_18_24"].max(),
    )

    colormap.caption = "Age 18–24 share"

    for _, row in df.iterrows():

        feature = hex_to_geojson(
            row["h3_9"],
            row["pred_age_18_24"],
        )

        folium.GeoJson(
            feature,
            style_function=lambda x: {
                "fillColor": colormap(x["properties"]["value"]),
                "color": "black",
                "weight": 0.2,
                "fillOpacity": 0.7,
            },
            tooltip=f"age_18_24: {row['pred_age_18_24']:.2f}",
        ).add_to(m)

    colormap.add_to(m)

    m.save(OUT_HTML)

    print("Saved map →", OUT_HTML)


if __name__ == "__main__":
    main()