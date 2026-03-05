from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import h3
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor


def main() -> None:
    targets = pd.read_parquet("data/kazan_targets_h3.parquet")
    feats = pd.read_parquet("data/kazan_hex_features_osm.parquet")
    df = targets.merge(feats, on="h3_9", how="inner")

    # y
    y = np.log1p(df["totals"].astype(float))

    # X
    drop_cols = [
        "segment_id", "totals",
        "male_share", "female_share",
        "age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_plus",
        "self_similarity", "ios_share",
        "lat", "lon",
        "h3_9",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Spatial split ---
    # группируем по "крупным" гексам (районы)
    parent_res = 7
    parents = df["h3_9"].apply(lambda hx: h3.cell_to_parent(hx, parent_res))

    unique_parents = parents.unique().to_numpy()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_parents)

    test_share = 0.2
    n_test = int(len(unique_parents) * test_share)
    test_parents = set(unique_parents[:n_test])

    is_test = parents.isin(test_parents)

    X_train, X_val = X[~is_test], X[is_test]
    y_train, y_val = y[~is_test], y[is_test]

    print("Train rows:", X_train.shape[0], "Val rows:", X_val.shape[0])
    print("Train parents:", len(set(parents[~is_test])), "Val parents:", len(set(parents[is_test])))

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.08,
        iterations=4000,
        loss_function="MAE",
        random_seed=42,
        verbose=200,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    pred_log = model.predict(X_val)
    mae_log = mean_absolute_error(y_val, pred_log)
    print("MAE on log1p(totals):", float(mae_log))

    y_true = np.expm1(y_val)
    y_pred = np.expm1(pred_log)
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)))
    print("MAPE on totals:", mape)

    fi = model.get_feature_importance()
    top = (
        pd.DataFrame({"feature": X.columns, "importance": fi})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    print("\nTop-20 features:")
    print(top.to_string(index=False))

    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"reach_catboost_spatial_parent{parent_res}.cbm"
    model.save_model(model_path)
    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()