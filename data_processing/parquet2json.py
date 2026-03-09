# read_res.py
import pandas as pd
df = pd.read_parquet("dataset/result/cot_results.parquet")
df.to_json("results/cot_predictions.json", orient="records", lines=True, force_ascii=False, indent=4)