from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import h3


@dataclass(frozen=True)
class SpatialLagConfig:
    h3_col: str = "h3_9"
    feature_cols: Optional[List[str]] = None
    prefix: str = "nb_mean_"


class SpatialLagFeatureExtractor:
    def __init__(self, config: Optional[SpatialLagConfig] = None):
        self.config = config or SpatialLagConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        h3_col = self.config.h3_col

        if h3_col not in df.columns:
            raise ValueError(f"Column '{h3_col}' not found in dataframe")

        if self.config.feature_cols is None:
            raise ValueError("feature_cols must be specified")

        work = df[[h3_col] + self.config.feature_cols].copy()
        work = work.drop_duplicates(subset=[h3_col]).set_index(h3_col)

        rows = []

        for hx in work.index:
            neighbors = list(h3.grid_disk(hx, 1))
            neighbors = [n for n in neighbors if n != hx]  # убираем сам hex

            existing_neighbors = [n for n in neighbors if n in work.index]

            row = {h3_col: hx}

            if existing_neighbors:
                nb_df = work.loc[existing_neighbors, self.config.feature_cols]
                for col in self.config.feature_cols:
                    row[f"{self.config.prefix}{col}"] = float(nb_df[col].mean())
            else:
                for col in self.config.feature_cols:
                    row[f"{self.config.prefix}{col}"] = np.nan

            rows.append(row)

        out = pd.DataFrame(rows)
        return out