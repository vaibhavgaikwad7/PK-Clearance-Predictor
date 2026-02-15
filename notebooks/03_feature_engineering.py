"""
Feature Engineering — PK Clearance Predictor
==============================================
Transforms raw pivoted demographics + interventions into ML-ready features.
Implements PK-specific calculations (BSA, estimated CrCl, dose normalization).
Joins groups ↔ interventions to create the analysis-ready dataset.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

RAW = "data/raw"
PROC = "data/processed"

# %%
# =============================================================
# 1. LOAD PROCESSED DATA
# =============================================================
print("=== Loading processed data ===")

df_groups = pd.read_csv(f"{PROC}/groups_wide.csv")
df_indiv = pd.read_csv(f"{PROC}/individuals_wide.csv")
df_interv = pd.read_csv(f"{RAW}/pkdb_interventions.csv")
df_caff = pd.read_csv(f"{RAW}/pkdb_caffeine_study_details.csv")

print(f"Groups wide:   {len(df_groups):,} rows × {df_groups.shape[1]} cols")
print(f"Indiv wide:    {len(df_indiv):,} rows × {df_indiv.shape[1]} cols")
print(f"Interventions: {len(df_interv):,} rows")
print(f"Caffeine studies: {len(df_caff):,}")

# %%
# =============================================================
# 2. PK-SPECIFIC FEATURE ENGINEERING FUNCTIONS
# =============================================================
print("\n=== Defining PK feature engineering functions ===")


def calc_bsa_dubois(weight_kg, height_cm):
    """Body Surface Area via DuBois formula (m²).
    BSA = 0.007184 × weight^0.425 × height^0.725
    Standard in pharmacokinetics for dose normalization.
    """
    mask = (weight_kg > 0) & (height_cm > 0)
    bsa = pd.Series(np.nan, index=weight_kg.index)
    bsa[mask] = 0.007184 * (weight_kg[mask] ** 0.425) * (height_cm[mask] ** 0.725)
    return bsa


def calc_bmi(weight_kg, height_cm):
    """BMI = weight(kg) / height(m)²"""
    mask = (weight_kg > 0) & (height_cm > 0)
    bmi = pd.Series(np.nan, index=weight_kg.index)
    height_m = height_cm[mask] / 100
    bmi[mask] = weight_kg[mask] / (height_m ** 2)
    return bmi


def calc_creatinine_clearance(age, weight_kg, sex, scr=1.0):
    """Cockcroft-Gault estimated creatinine clearance (mL/min).
    CrCl = [(140 - age) × weight] / (72 × SCr)
    Multiply by 0.85 for females.
    Default SCr=1.0 mg/dL when not available (population average).
    """
    crcl = ((140 - age) * weight_kg) / (72 * scr)
    # Apply female correction
    is_female = sex.str.lower().isin(['f', 'female', 'F'])
    crcl[is_female] = crcl[is_female] * 0.85
    return crcl


def calc_ideal_body_weight(height_cm, sex):
    """Ideal Body Weight via Devine formula (kg).
    Male:   IBW = 50 + 2.3 × (height_inches - 60)
    Female: IBW = 45.5 + 2.3 × (height_inches - 60)
    """
    height_in = height_cm / 2.54
    ibw = pd.Series(np.nan, index=height_cm.index)
    is_female = sex.str.lower().isin(['f', 'female', 'F'])
    is_male = ~is_female

    mask_m = is_male & height_cm.notna()
    mask_f = is_female & height_cm.notna()

    ibw[mask_m] = 50 + 2.3 * (height_in[mask_m] - 60)
    ibw[mask_f] = 45.5 + 2.3 * (height_in[mask_f] - 60)
    return ibw


def encode_binary(series, positive_values):
    """Encode categorical as binary (1 = positive, 0 = negative, NaN = missing)."""
    result = pd.Series(np.nan, index=series.index)
    mask = series.notna()
    result[mask] = series[mask].str.lower().isin([v.lower() for v in positive_values]).astype(int)
    return result


print("Defined: calc_bsa_dubois, calc_bmi, calc_creatinine_clearance,")
print("         calc_ideal_body_weight, encode_binary")

# %%
# =============================================================
# 3. APPLY FEATURES TO GROUPS
# =============================================================
print("\n=== Applying features to groups ===")

df_g = df_groups.copy()

# 3a. Calculate BMI where missing but height + weight available
if 'bmi' not in df_g.columns:
    df_g['bmi'] = np.nan
if 'height' in df_g.columns:
    missing_bmi = df_g['bmi'].isna() & df_g['weight'].notna() & df_g['height'].notna()
    df_g.loc[missing_bmi, 'bmi'] = calc_bmi(
        df_g.loc[missing_bmi, 'weight'],
        df_g.loc[missing_bmi, 'height']
    )
    print(f"  BMI: filled {missing_bmi.sum()} missing values from height+weight")

# 3b. Body Surface Area
if 'height' in df_g.columns:
    df_g['bsa'] = calc_bsa_dubois(
        df_g['weight'].fillna(0),
        df_g['height'].fillna(0)
    )
    df_g.loc[df_g['bsa'] == 0, 'bsa'] = np.nan
    print(f"  BSA: calculated for {df_g['bsa'].notna().sum()} groups")

# 3c. Estimated creatinine clearance
has_crcl = df_g['age'].notna() & df_g['weight'].notna()
if 'sex' in df_g.columns:
    has_crcl = has_crcl & df_g['sex'].notna()
    df_g['est_crcl'] = np.nan
    df_g.loc[has_crcl, 'est_crcl'] = calc_creatinine_clearance(
        df_g.loc[has_crcl, 'age'],
        df_g.loc[has_crcl, 'weight'],
        df_g.loc[has_crcl, 'sex']
    )
    print(f"  CrCl: estimated for {df_g['est_crcl'].notna().sum()} groups")

# 3d. Ideal body weight
if 'height' in df_g.columns and 'sex' in df_g.columns:
    has_ibw = df_g['height'].notna() & df_g['sex'].notna()
    df_g['ibw'] = np.nan
    df_g.loc[has_ibw, 'ibw'] = calc_ideal_body_weight(
        df_g.loc[has_ibw, 'height'],
        df_g.loc[has_ibw, 'sex']
    )
    print(f"  IBW: calculated for {df_g['ibw'].notna().sum()} groups")

# 3e. Binary encodings
if 'smoking' in df_g.columns:
    df_g['is_smoker'] = encode_binary(df_g['smoking'], ['yes', 'true', 'smoker', 'Y'])
    print(f"  Smoker: encoded for {df_g['is_smoker'].notna().sum()} groups")

if 'healthy' in df_g.columns:
    df_g['is_healthy'] = encode_binary(df_g['healthy'], ['yes', 'true', 'Y', 'healthy'])
    print(f"  Healthy: encoded for {df_g['is_healthy'].notna().sum()} groups")

if 'oral contraceptives' in df_g.columns:
    df_g['on_oc'] = encode_binary(df_g['oral contraceptives'], ['yes', 'true', 'Y'])
    print(f"  OC use: encoded for {df_g['on_oc'].notna().sum()} groups")

if 'sex' in df_g.columns:
    df_g['is_female'] = encode_binary(df_g['sex'], ['F', 'female', 'f'])
    print(f"  Female: encoded for {df_g['is_female'].notna().sum()} groups")

# 3f. Age categories (clinical standard)
df_g['age_category'] = pd.cut(
    df_g['age'],
    bins=[0, 18, 40, 65, 100],
    labels=['pediatric', 'young_adult', 'middle_aged', 'elderly'],
    right=False
)
print(f"  Age category: {df_g['age_category'].notna().sum()} groups categorized")

# 3g. BMI categories (WHO standard)
if 'bmi' in df_g.columns:
    df_g['bmi_category'] = pd.cut(
        df_g['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['underweight', 'normal', 'overweight', 'obese'],
        right=False
    )
    print(f"  BMI category: {df_g['bmi_category'].notna().sum()} groups categorized")

print(f"\nFinal group features: {df_g.shape[1]} columns")
print(f"Columns: {list(df_g.columns)}")

# %%
# =============================================================
# 4. APPLY FEATURES TO INDIVIDUALS
# =============================================================
print("\n=== Applying features to individuals ===")

df_i = df_indiv.copy()

# BMI
if 'bmi' not in df_i.columns:
    df_i['bmi'] = np.nan
if 'height' in df_i.columns:
    missing = df_i['bmi'].isna() & df_i['weight'].notna() & df_i['height'].notna()
    df_i.loc[missing, 'bmi'] = calc_bmi(df_i.loc[missing, 'weight'], df_i.loc[missing, 'height'])
    print(f"  BMI: filled {missing.sum()} values")

# BSA
if 'height' in df_i.columns:
    df_i['bsa'] = calc_bsa_dubois(df_i['weight'].fillna(0), df_i['height'].fillna(0))
    df_i.loc[df_i['bsa'] == 0, 'bsa'] = np.nan
    print(f"  BSA: {df_i['bsa'].notna().sum()} individuals")

# CrCl
if 'sex' in df_i.columns:
    has = df_i['age'].notna() & df_i['weight'].notna() & df_i['sex'].notna()
    df_i['est_crcl'] = np.nan
    df_i.loc[has, 'est_crcl'] = calc_creatinine_clearance(
        df_i.loc[has, 'age'], df_i.loc[has, 'weight'], df_i.loc[has, 'sex']
    )
    print(f"  CrCl: {df_i['est_crcl'].notna().sum()} individuals")

# Binary encodings
if 'sex' in df_i.columns:
    df_i['is_female'] = encode_binary(df_i['sex'], ['F', 'female', 'f'])

if 'smoking' in df_i.columns:
    df_i['is_smoker'] = encode_binary(df_i['smoking'], ['yes', 'true', 'smoker', 'Y'])

# Age + BMI categories
df_i['age_category'] = pd.cut(df_i['age'], bins=[0, 18, 40, 65, 100],
                               labels=['pediatric', 'young_adult', 'middle_aged', 'elderly'],
                               right=False)

print(f"\nFinal individual features: {df_i.shape[1]} columns")

# %%
# =============================================================
# 5. BUILD CAFFEINE INTERVENTION FEATURES
# =============================================================
print("\n=== Caffeine intervention features ===")

caff_sids = set(df_caff['sid'].tolist())

# Filter interventions to caffeine studies
df_caff_interv = df_interv[
    (df_interv['study_sid'].isin(caff_sids)) &
    (df_interv['measurement_type'] == 'dosing')
].copy()

print(f"Caffeine dosing records: {len(df_caff_interv):,}")
print(f"\nSubstances in caffeine studies:")
print(df_caff_interv['substance'].value_counts().head(10))
print(f"\nRoutes:")
print(df_caff_interv['route'].value_counts())
print(f"\nApplication types:")
print(df_caff_interv['application'].value_counts())

# Create intervention summary per study
interv_summary = df_caff_interv.groupby('study_sid').agg(
    n_interventions=('intervention_pk', 'nunique'),
    primary_substance=('substance', lambda x: x.mode().iloc[0] if len(x) > 0 else None),
    primary_route=('route', lambda x: x.mode().iloc[0] if len(x) > 0 else None),
    dose_value=('value', 'first'),
    dose_unit=('unit', 'first'),
    application_type=('application', 'first'),
).reset_index()

print(f"\nIntervention summary: {len(interv_summary)} studies")
print(interv_summary.head())

# %%
# =============================================================
# 6. JOIN: Caffeine groups + interventions → analysis dataset
# =============================================================
print("\n=== Building caffeine analysis dataset ===")

# Filter groups to caffeine studies
df_caff_groups = df_g[df_g['study_sid'].isin(caff_sids)].copy()
print(f"Caffeine groups with features: {len(df_caff_groups):,}")

# Join with intervention summary
df_analysis = pd.merge(
    df_caff_groups,
    interv_summary,
    on='study_sid',
    how='left'
)

print(f"Analysis dataset: {len(df_analysis):,} rows × {df_analysis.shape[1]} cols")
print(f"\nColumn overview:")
for col in df_analysis.columns:
    non_null = df_analysis[col].notna().sum()
    pct = non_null / len(df_analysis) * 100
    dtype = df_analysis[col].dtype
    print(f"  {col:30s} {dtype:>10s}  {non_null:>5,} ({pct:5.1f}%)")

df_analysis.to_csv(f"{PROC}/caffeine_analysis_dataset.csv", index=False)
print(f"\nSaved to {PROC}/caffeine_analysis_dataset.csv")

# %%
# =============================================================
# 7. VISUALIZE ENGINEERED FEATURES
# =============================================================
print("\n=== Feature Distribution Plots ===")

eng_features = ['bsa', 'est_crcl', 'ibw', 'bmi']
available = [f for f in eng_features if f in df_caff_groups.columns and df_caff_groups[f].notna().sum() > 5]

if available:
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for i, feat in enumerate(available):
        data = df_caff_groups[feat].dropna()
        axes[i].hist(data, bins=25, color=colors[i % 4], edgecolor='white', alpha=0.8)
        axes[i].set_title(f'{feat.upper()}\n(n={len(data)}, μ={data.mean():.1f}, σ={data.std():.1f})')
        axes[i].set_xlabel(feat)

    plt.suptitle('Engineered Features — Caffeine Study Groups', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PROC}/fig_engineered_features.png", dpi=150, bbox_inches='tight')
    plt.show()

# %%
# =============================================================
# 8. SMOKING × SEX INTERACTION IN CAFFEINE STUDIES
#    (The PK-DB paper showed smoking + OC strongly affect caffeine clearance)
# =============================================================
print("\n=== Smoking × Sex Interaction (Caffeine Groups) ===")

if 'is_smoker' in df_caff_groups.columns and 'is_female' in df_caff_groups.columns:
    cross = pd.crosstab(
        df_caff_groups['is_smoker'].map({0: 'Non-smoker', 1: 'Smoker'}),
        df_caff_groups['is_female'].map({0: 'Male', 1: 'Female'}),
        margins=True
    )
    print(cross)

    if 'age' in df_caff_groups.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Age by smoking
        mask_s = df_caff_groups['is_smoker'].notna() & df_caff_groups['age'].notna()
        if mask_s.sum() > 5:
            df_caff_groups.loc[mask_s].boxplot(column='age', by='is_smoker', ax=axes[0])
            axes[0].set_title('Age by Smoking Status')
            axes[0].set_xlabel('Smoker (0=No, 1=Yes)')
            axes[0].set_ylabel('Age')
            plt.sca(axes[0])
            plt.title('Age by Smoking Status')

        # Weight by sex
        mask_w = df_caff_groups['is_female'].notna() & df_caff_groups['weight'].notna()
        if mask_w.sum() > 5:
            df_caff_groups.loc[mask_w].boxplot(column='weight', by='is_female', ax=axes[1])
            axes[1].set_title('Weight by Sex')
            axes[1].set_xlabel('Female (0=No, 1=Yes)')
            axes[1].set_ylabel('Weight (kg)')
            plt.sca(axes[1])
            plt.title('Weight by Sex')

        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f"{PROC}/fig_smoking_sex_interaction.png", dpi=150, bbox_inches='tight')
        plt.show()

# %%
# =============================================================
# 9. SAVE REUSABLE FEATURE FUNCTIONS TO src/
# =============================================================
print("\n=== Saving feature functions to src/transformation/ ===")

func_code = '''"""
PK Feature Engineering Functions
=================================
Pharmacokinetics-specific calculations for patient covariates.
Used in both training pipeline and prediction API.
"""
import numpy as np
import pandas as pd


def calc_bsa_dubois(weight_kg, height_cm):
    """Body Surface Area via DuBois formula (m²).
    BSA = 0.007184 × weight^0.425 × height^0.725
    """
    mask = (weight_kg > 0) & (height_cm > 0)
    bsa = pd.Series(np.nan, index=weight_kg.index)
    bsa[mask] = 0.007184 * (weight_kg[mask] ** 0.425) * (height_cm[mask] ** 0.725)
    return bsa


def calc_bmi(weight_kg, height_cm):
    """BMI = weight(kg) / height(m)²"""
    mask = (weight_kg > 0) & (height_cm > 0)
    bmi = pd.Series(np.nan, index=weight_kg.index)
    height_m = height_cm[mask] / 100
    bmi[mask] = weight_kg[mask] / (height_m ** 2)
    return bmi


def calc_creatinine_clearance(age, weight_kg, sex, scr=1.0):
    """Cockcroft-Gault estimated creatinine clearance (mL/min).
    CrCl = [(140 - age) × weight] / (72 × SCr)
    Multiply by 0.85 for females.
    """
    crcl = ((140 - age) * weight_kg) / (72 * scr)
    is_female = sex.str.lower().isin([\'f\', \'female\'])
    crcl[is_female] = crcl[is_female] * 0.85
    return crcl


def calc_ideal_body_weight(height_cm, sex):
    """Ideal Body Weight via Devine formula (kg).
    Male:   IBW = 50 + 2.3 × (height_inches - 60)
    Female: IBW = 45.5 + 2.3 × (height_inches - 60)
    """
    height_in = height_cm / 2.54
    ibw = pd.Series(np.nan, index=height_cm.index)
    is_female = sex.str.lower().isin([\'f\', \'female\'])
    mask_m = ~is_female & height_cm.notna()
    mask_f = is_female & height_cm.notna()
    ibw[mask_m] = 50 + 2.3 * (height_in[mask_m] - 60)
    ibw[mask_f] = 45.5 + 2.3 * (height_in[mask_f] - 60)
    return ibw


def encode_binary(series, positive_values):
    """Encode categorical as binary (1/0/NaN)."""
    result = pd.Series(np.nan, index=series.index)
    mask = series.notna()
    result[mask] = series[mask].str.lower().isin(
        [v.lower() for v in positive_values]
    ).astype(int)
    return result


def add_age_category(age_series):
    """Clinical age categories."""
    return pd.cut(age_series, bins=[0, 18, 40, 65, 100],
                  labels=[\'pediatric\', \'young_adult\', \'middle_aged\', \'elderly\'],
                  right=False)


def add_bmi_category(bmi_series):
    """WHO BMI categories."""
    return pd.cut(bmi_series, bins=[0, 18.5, 25, 30, 100],
                  labels=[\'underweight\', \'normal\', \'overweight\', \'obese\'],
                  right=False)
'''

os.makedirs("src/transformation", exist_ok=True)
with open("src/transformation/pk_calculations.py", "w") as f:
    f.write(func_code)

print("Saved: src/transformation/pk_calculations.py")

# %%
# =============================================================
# 10. FINAL FEATURE SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("  FEATURE ENGINEERING SUMMARY")
print("=" * 60)

print(f"\n  Engineered features added:")
print(f"    bsa          — Body Surface Area (DuBois)")
print(f"    est_crcl     — Estimated Creatinine Clearance (Cockcroft-Gault)")
print(f"    ibw          — Ideal Body Weight (Devine)")
print(f"    bmi          — BMI (calculated where missing)")
print(f"    is_smoker    — Binary smoking status")
print(f"    is_healthy   — Binary health status")
print(f"    is_female    — Binary sex encoding")
print(f"    on_oc        — Binary oral contraceptive use")
print(f"    age_category — Pediatric/Young/Middle/Elderly")
print(f"    bmi_category — Underweight/Normal/Overweight/Obese")

print(f"\n  Output files:")
print(f"    {PROC}/caffeine_analysis_dataset.csv — Caffeine groups + interventions joined")
print(f"    src/transformation/pk_calculations.py — Reusable PK functions")

print(f"\n  Caffeine analysis dataset shape: {df_analysis.shape}")

print(f"\n{'='*60}")
print("  READY FOR ML MODELING")
print(f"{'='*60}")
print("""
Next steps:
  1. Get TDC clearance data (via Colab or conda env with Python 3.10)
  2. Train XGBoost/LASSO/RF on clearance prediction
  3. SHAP analysis for feature importance
  4. Power BI dashboard from caffeine_analysis_dataset.csv
""")