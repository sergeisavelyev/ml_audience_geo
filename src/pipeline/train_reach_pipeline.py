from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor


def main() -> None:
    # --- Пути к данным ---
    targets_path = Path("data/kazan_targets_h3.parquet")
    features_path = Path("data/kazan_hex_features_osm.parquet")

    # --- 1) Читаем данные ---
    targets = pd.read_parquet(targets_path)
    feats = pd.read_parquet(features_path)

    # --- 2) Merge: hex -> (features + target) ---
    df = targets.merge(feats, on="h3_9", how="inner")

    # --- 3) Формируем y (таргет) ---
    # totals — это reach, распределение обычно "косое", поэтому берём log1p
    y = np.log1p(df["totals"].astype(float))

    # --- 4) Формируем X (признаки) ---
    # Убираем колонки-таргеты и служебные
    drop_cols = [
        "segment_id", "totals",
        "male_share", "female_share",
        "age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_plus",
        "self_similarity", "ios_share",
        "lat", "lon",  # центроиды сегментов (не обязательно)
        "h3_9",        # это id, не признак
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # На всякий случай проверим что нет NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- 5) Делим на train/val ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 6) Обучаем модель ---
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.08,
        iterations=2000,
        loss_function="MAE",
        random_seed=42,
        verbose=200,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    # --- 7) Оценка качества ---
    pred_log = model.predict(X_val)
    mae_log = mean_absolute_error(y_val, pred_log)
    print("MAE on log1p(totals):", float(mae_log))

    # Переводим обратно в "totals" и считаем MAPE (ошибка в %)
    y_true = np.expm1(y_val)
    y_pred = np.expm1(pred_log)

    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)))
    print("MAPE on totals:", mape)

    # --- 8) Важность признаков ---
    fi = model.get_feature_importance()
    top = (
        pd.DataFrame({"feature": X.columns, "importance": fi})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    print("\nTop-20 features:")
    print(top.to_string(index=False))

    # --- 9) (Опционально) сохраняем модель ---
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "reach_catboost.cbm"
    model.save_model(model_path)
    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()