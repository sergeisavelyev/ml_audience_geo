import pandas as pd

m = pd.read_parquet("data/kazan_reach_map_h3_9.parquet")
print(m["pred_totals"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
print("min hex:", m.loc[m["pred_totals"].idxmin(), ["h3_9","pred_totals","lat_c","lon_c"]].to_dict())
print("max hex:", m.loc[m["pred_totals"].idxmax(), ["h3_9","pred_totals","lat_c","lon_c"]].to_dict())