"""Quick re-run: build caffeine analysis dataset + save feature functions."""
import pandas as pd
import numpy as np
import os, sys

# Add project root to path
sys.path.insert(0, ".")

RAW = "data/raw"
PROC = "data/processed"

# Load
df_groups = pd.read_csv(f"{PROC}/groups_wide.csv")
df_interv = pd.read_csv(f"{RAW}/pkdb_interventions.csv")
df_caff = pd.read_csv(f"{RAW}/pkdb_caffeine_study_details.csv")

# === Re-apply feature engineering to groups ===
df_g = df_groups.copy()

def calc_bsa(w, h):
    m = (w > 0) & (h > 0)
    r = pd.Series(np.nan, index=w.index)
    r[m] = 0.007184 * (w[m] ** 0.425) * (h[m] ** 0.725)
    return r

def calc_crcl(age, w, sex, scr=1.0):
    c = ((140 - age) * w) / (72 * scr)
    c[sex.str.lower().isin(['f', 'female'])] *= 0.85
    return c

def encode_bin(s, pos):
    r = pd.Series(np.nan, index=s.index)
    m = s.notna()
    r[m] = s[m].str.lower().isin([v.lower() for v in pos]).astype(int)
    return r

# BMI fill
if 'height' in df_g.columns:
    m = df_g['bmi'].isna() & df_g['weight'].notna() & df_g['height'].notna()
    h = df_g.loc[m, 'height'] / 100
    df_g.loc[m, 'bmi'] = df_g.loc[m, 'weight'] / (h ** 2)

# BSA
if 'height' in df_g.columns:
    df_g['bsa'] = calc_bsa(df_g['weight'].fillna(0), df_g['height'].fillna(0))
    df_g.loc[df_g['bsa'] == 0, 'bsa'] = np.nan

# CrCl
if 'sex' in df_g.columns:
    m = df_g['age'].notna() & df_g['weight'].notna() & df_g['sex'].notna()
    df_g['est_crcl'] = np.nan
    df_g.loc[m, 'est_crcl'] = calc_crcl(df_g.loc[m, 'age'], df_g.loc[m, 'weight'], df_g.loc[m, 'sex'])

# IBW
if 'height' in df_g.columns and 'sex' in df_g.columns:
    h_in = df_g['height'] / 2.54
    df_g['ibw'] = np.nan
    m_m = ~df_g['sex'].str.lower().isin(['f', 'female']) & df_g['height'].notna()
    m_f = df_g['sex'].str.lower().isin(['f', 'female']) & df_g['height'].notna()
    df_g.loc[m_m, 'ibw'] = 50 + 2.3 * (h_in[m_m] - 60)
    df_g.loc[m_f, 'ibw'] = 45.5 + 2.3 * (h_in[m_f] - 60)

# Binary encodings
if 'smoking' in df_g.columns:
    df_g['is_smoker'] = encode_bin(df_g['smoking'], ['yes', 'true', 'smoker', 'Y'])
if 'healthy' in df_g.columns:
    df_g['is_healthy'] = encode_bin(df_g['healthy'], ['yes', 'true', 'Y', 'healthy'])
if 'oral contraceptives' in df_g.columns:
    df_g['on_oc'] = encode_bin(df_g['oral contraceptives'], ['yes', 'true', 'Y'])
if 'sex' in df_g.columns:
    df_g['is_female'] = encode_bin(df_g['sex'], ['F', 'female', 'f'])

df_g['age_category'] = pd.cut(df_g['age'], bins=[0, 18, 40, 65, 100],
                               labels=['pediatric', 'young_adult', 'middle_aged', 'elderly'], right=False)
if 'bmi' in df_g.columns:
    df_g['bmi_category'] = pd.cut(df_g['bmi'], bins=[0, 18.5, 25, 30, 100],
                                   labels=['underweight', 'normal', 'overweight', 'obese'], right=False)

# === Caffeine interventions ===
caff_sids = set(df_caff['sid'].tolist())
df_caff_interv = df_interv[(df_interv['study_sid'].isin(caff_sids)) & (df_interv['measurement_type'] == 'dosing')]

interv_summary = df_caff_interv.groupby('study_sid').agg(
    n_interventions=('intervention_pk', 'nunique'),
    primary_substance=('substance', lambda x: x.mode().iloc[0] if len(x) > 0 else None),
    primary_route=('route', lambda x: x.mode().iloc[0] if len(x) > 0 else None),
    dose_value=('value', 'first'),
    dose_unit=('unit', 'first'),
    application_type=('application', 'first'),
).reset_index()

# === Join ===
df_caff_groups = df_g[df_g['study_sid'].isin(caff_sids)]
df_analysis = pd.merge(df_caff_groups, interv_summary, on='study_sid', how='left')

# === Save ===
df_analysis.to_csv(f"{PROC}/caffeine_analysis_dataset.csv", index=False)
print(f"Saved: {PROC}/caffeine_analysis_dataset.csv")
print(f"Shape: {df_analysis.shape}")

# Column overview
print(f"\nColumn overview:")
for col in df_analysis.columns:
    n = df_analysis[col].notna().sum()
    pct = n / len(df_analysis) * 100
    print(f"  {col:30s} {str(df_analysis[col].dtype):>10s}  {n:>5} ({pct:5.1f}%)")