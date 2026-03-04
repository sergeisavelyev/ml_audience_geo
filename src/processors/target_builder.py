class TargetBuilder:

    def build(self, audience_df, polygon_df):

        df = audience_df.merge(
            polygon_df,
            on="segment_id",
            how="inner"
        )

        return df