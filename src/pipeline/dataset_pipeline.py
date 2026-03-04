from pathlib import Path

from src.loaders.yandex_loader import YandexAudienceLoader
from src.loaders.polygon_loader import PolygonLoader

from src.processors.target_builder import TargetBuilder
from src.processors.hex_builder import HexBuilder


def main():

    audience_path = Path("data/audience.json")
    polygons_path = Path("data/polygons.json")

    audience_df = YandexAudienceLoader(audience_path).to_dataframe()

    polygons_df = PolygonLoader(polygons_path).to_centroid_dataframe()

    dataset = TargetBuilder().build(
        audience_df,
        polygons_df
    )

    dataset = HexBuilder().add_hex(dataset)

    print(dataset.head())

    dataset.to_parquet(
        "data/kazan_targets_h3.parquet",
        index=False
    )


if __name__ == "__main__":
    main()