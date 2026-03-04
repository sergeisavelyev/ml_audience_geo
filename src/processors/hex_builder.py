import h3


class HexBuilder:

    def __init__(self, resolution=9):
        self.resolution = resolution

    def add_hex(self, df):

        df["h3_9"] = df.apply(
            lambda r: h3.latlng_to_cell(
                r["lat"],
                r["lon"],
                self.resolution
            ),
            axis=1
        )

        return df