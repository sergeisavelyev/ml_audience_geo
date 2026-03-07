берем небольшое число сегментов (у нас Казань: ~937)

учим модель находить связь “городская среда → аудитория”

потом предсказываем метрики для любой точки (в виде карты гексов H3)

Что лежит в папках

data/ — данные и результаты
    audience.json — ответы Яндекс Аудиторий по сегментам (reach/пол/возраст/интересы)
    polygons.json — полигоны (гексагоны) сегментов с координатами
    kazan_targets_h3.parquet — “таргеты” для обучения (1 строка = 1 сегмент/гекс + метрики)
    kazan_hex_features_osm.parquet — признаки городской среды для каждого H3-гекса (POI/дороги/landuse)
    kazan_reach_map_h3_9.parquet — карта Казани: для каждого гекса предсказанный reach
    kazan_reach_map_h3_9.geojson — то же самое, но как слой карты (полигоны гексов)
    models/ — сохраненные модели:
        reach_catboost.cbm — модель reach (обычная проверка)
        reach_catboost_spatial_parent7.cbm — модель reach (честная spatial-проверка)

src/ — код
    src/loaders/ — загрузчики “сырых” файлов (json → таблицы)
    src/features/ — извлечение признаков из OpenStreetMap (OSM → фичи для hex)
    src/pipeline/ — “кнопки запуска”: сценарии, которые запускают весь шаг целиком

Как устроен pipeline (по смыслу)
Шаг А. “Сделать датасет для обучения”

Мы превращаем ответы Яндекс Аудиторий в таблицу “таргетов”:
где находится сегмент (lat/lon, h3)
какие метрики аудитории (totals, пол, возраст, интересы)

Запуск:
python -m src.pipeline.dataset_pipeline

Результат:
data/kazan_targets_h3.parquet

Шаг B. “Собрать признаки городской среды”

Мы для каждого H3-гекса строим признаки из OpenStreetMap:

POI (кафе, банки, магазины…)
дороги (типы дорог + длины в метрах)
landuse (жилое/торговое/промышленное…)

Запуск:

python -m src.pipeline.features_pipeline

Результат:
data/kazan_hex_features_osm.parquet

Шаг C. “Обучить модель reach”

Мы обучаем модель, которая по признакам OSM предсказывает totals (reach).
Запуск (обычная проверка):
python -m src.pipeline.train_reach_pipeline

Результат:
в консоли метрики качества
модель сохраняется в data/models/reach_catboost.cbm

Шаг D. “Честно проверить качество для геоданных (Spatial split)”
Это важный шаг именно для гео: мы проверяем модель не на случайных точках, а на “других районах города”.

Запуск:

python -m src.pipeline.train_reach_spatial_split_pipeline

Результат:
честные метрики

модель сохраняется в data/models/reach_catboost_spatial_parent7.cbm

Шаг E. “Сделать настоящую карту Казани по всем гексам”

Мы строим контур города (из polygons.json), заполняем его гексами H3 (res=9), считаем для каждого гекса признаки OSM и прогоняем модель → получаем “покрашенную” карту reach.

Запуск:
python -m src.pipeline.predict_reach_map_kazan

Результат:
data/kazan_reach_map_h3_9.parquet

Шаг F. “Проверить карту цифрами”

Быстро смотрим распределение, min/max и координаты — чтобы убедиться, что всё похоже на правду.

Запуск:
python -m src.pipeline.check_reach_map

Шаг G. “Сделать слой карты (GeoJSON)”

Преобразуем H3-гексы в реальные полигоны и сохраняем GeoJSON, который можно открыть в Kepler.gl или QGIS.

Запуск:
python -m src.pipeline.export_reach_map_geojson

Результат:
data/kazan_reach_map_h3_9.geojson

Как посмотреть карту

Самый простой способ:
открыть Kepler.gl

загрузить data/kazan_reach_map_h3_9.geojson
слой Polygon → “Fill Color” по pred_totals

Самая короткая последовательность “с нуля”

Если у тебя уже есть audience.json и polygons.json:

python -m src.pipeline.dataset_pipeline
python -m src.pipeline.features_pipeline
python -m src.pipeline.train_reach_spatial_split_pipeline
python -m src.pipeline.predict_reach_map_kazan
python -m src.pipeline.export_reach_map_geojson
Что важно помнить (простыми словами)

targets = “что Яндекс сказал про аудиторию” (это то, чему учим)
features = “что вокруг на карте” (это то, по чему модель делает вывод)
model = “мост” между features и targets
map = применение модели на большой сетке гексов