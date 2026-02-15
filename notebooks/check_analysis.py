# Just re-run from the analysis dataset save onwards
# The data is already built correctly, just the print failed

import pandas as pd
PROC = "data/processed"
df_analysis = pd.read_csv(f"{PROC}/caffeine_analysis_dataset.csv")

print(f"Analysis dataset: {len(df_analysis):,} rows × {df_analysis.shape[1]} cols")
print(f"\nColumn overview:")
for col in df_analysis.columns:
    non_null = df_analysis[col].notna().sum()
    pct = non_null / len(df_analysis) * 100
    dtype = str(df_analysis[col].dtype)
    print(f"  {col:30s} {dtype:>10s}  {non_null:>5} ({pct:5.1f}%)")