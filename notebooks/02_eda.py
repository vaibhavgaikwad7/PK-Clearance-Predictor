"""
Exploratory Data Analysis — PK Clearance Predictor
====================================================
Transforms long-format PK-DB data into analysis-ready wide format,
explores demographics, interventions, and caffeine study patterns.
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
plt.rcParams['figure.figsize'] = (12, 6)

RAW = "data/raw"
PROC = "data/processed"
os.makedirs(PROC, exist_ok=True)

# %%
# =============================================================
# 1. LOAD RAW DATA
# =============================================================
print("=== Loading raw data ===")

df_studies = pd.read_csv(f"{RAW}/pkdb_studies.csv")
df_groups = pd.read_csv(f"{RAW}/pkdb_groups.csv")
df_indiv = pd.read_csv(f"{RAW}/pkdb_individuals.csv")
df_interv = pd.read_csv(f"{RAW}/pkdb_interventions.csv")
df_subs = pd.read_csv(f"{RAW}/pkdb_substance_stats.csv")
df_caff = pd.read_csv(f"{RAW}/pkdb_caffeine_study_details.csv")

print(f"Studies:       {len(df_studies):,}")
print(f"Groups:        {len(df_groups):,}")
print(f"Individuals:   {len(df_indiv):,}")
print(f"Interventions: {len(df_interv):,}")
print(f"Substances:    {len(df_subs):,}")
print(f"Caffeine studies: {len(df_caff):,}")

# %%
# =============================================================
# 2. PIVOT GROUPS: Long → Wide (one row per group)
#    Each row currently = one measurement_type for one group
#    We need: one row per group with age, weight, sex, etc. as columns
# =============================================================
print("\n=== Pivoting group demographics ===")

# Check what measurement types exist
print("Group measurement types:")
print(df_groups['measurement_type'].value_counts().head(20))

# Separate numeric vs categorical measurements
numeric_types = ['age', 'weight', 'bmi', 'height', 'concentration']
choice_types = ['sex', 'healthy', 'species', 'smoking', 'ethnicity',
                'medication', 'overnight fast', 'disease',
                'abstinence', 'oral contraceptives', 'abstinence medication']

# Pivot numeric: use 'mean' column
df_grp_num = df_groups[df_groups['measurement_type'].isin(numeric_types)].copy()
grp_numeric = df_grp_num.pivot_table(
    index=['study_sid', 'study_name', 'group_pk', 'group_name', 'group_count'],
    columns='measurement_type',
    values='mean',
    aggfunc='first'
).reset_index()
grp_numeric.columns.name = None

# Pivot categorical: use 'choice' column
df_grp_cat = df_groups[df_groups['measurement_type'].isin(choice_types)].copy()
grp_choice = df_grp_cat.pivot_table(
    index=['study_sid', 'study_name', 'group_pk', 'group_name', 'group_count'],
    columns='measurement_type',
    values='choice',
    aggfunc='first'
).reset_index()
grp_choice.columns.name = None

# Merge numeric + categorical
df_groups_wide = pd.merge(grp_numeric, grp_choice,
                          on=['study_sid', 'study_name', 'group_pk', 'group_name', 'group_count'],
                          how='outer')

print(f"Groups wide: {len(df_groups_wide):,} rows × {len(df_groups_wide.columns)} columns")
print(f"Columns: {list(df_groups_wide.columns)}")
print(f"\nSample:")
print(df_groups_wide.head())

df_groups_wide.to_csv(f"{PROC}/groups_wide.csv", index=False)
print(f"Saved to {PROC}/groups_wide.csv")

# %%
# =============================================================
# 3. PIVOT INDIVIDUALS: Long → Wide (one row per individual)
# =============================================================
print("\n=== Pivoting individual demographics ===")

print("Individual measurement types:")
print(df_indiv['measurement_type'].value_counts().head(20))

indiv_numeric_types = ['age', 'weight', 'bmi', 'height']
indiv_choice_types = ['sex', 'healthy', 'species', 'smoking', 'ethnicity',
                      'disease', 'medication', 'cyp2d6 phenotype',
                      'cyp2d6 genotype', 'child-pugh score',
                      'abstinence medication']

# Pivot numeric
df_ind_num = df_indiv[df_indiv['measurement_type'].isin(indiv_numeric_types)].copy()
ind_numeric = df_ind_num.pivot_table(
    index=['study_sid', 'study_name', 'individual_pk', 'individual_name', 'individual_group_pk'],
    columns='measurement_type',
    values='value',  # individuals use 'value' not 'mean'
    aggfunc='first'
).reset_index()
ind_numeric.columns.name = None

# Pivot categorical
df_ind_cat = df_indiv[df_indiv['measurement_type'].isin(indiv_choice_types)].copy()
ind_choice = df_ind_cat.pivot_table(
    index=['study_sid', 'study_name', 'individual_pk', 'individual_name', 'individual_group_pk'],
    columns='measurement_type',
    values='choice',
    aggfunc='first'
).reset_index()
ind_choice.columns.name = None

# Merge
df_indiv_wide = pd.merge(ind_numeric, ind_choice,
                         on=['study_sid', 'study_name', 'individual_pk',
                             'individual_name', 'individual_group_pk'],
                         how='outer')

print(f"Individuals wide: {len(df_indiv_wide):,} rows × {len(df_indiv_wide.columns)} columns")
print(f"Columns: {list(df_indiv_wide.columns)}")
print(f"\nNull rates:")
null_pct = (df_indiv_wide.isnull().sum() / len(df_indiv_wide) * 100).round(1)
print(null_pct.to_string())

df_indiv_wide.to_csv(f"{PROC}/individuals_wide.csv", index=False)
print(f"\nSaved to {PROC}/individuals_wide.csv")

# %%
# =============================================================
# 4. INTERVENTION ANALYSIS — Dosing patterns
# =============================================================
print("\n=== Intervention Analysis ===")

# Filter to dosing interventions only
df_doses = df_interv[df_interv['measurement_type'] == 'dosing'].copy()
print(f"Dosing records: {len(df_doses):,}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Top substances
top_subs = df_doses['substance'].value_counts().head(15)
top_subs.plot(kind='barh', ax=axes[0], color=sns.color_palette("muted", 15))
axes[0].set_title('Top 15 Substances by Dosing Records')
axes[0].set_xlabel('Count')
axes[0].invert_yaxis()

# Routes
route_counts = df_doses['route'].value_counts()
route_counts.plot(kind='bar', ax=axes[1], color=sns.color_palette("Set2"))
axes[1].set_title('Administration Routes')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

# Application types
app_counts = df_doses['application'].value_counts()
app_counts.plot(kind='bar', ax=axes[2], color=sns.color_palette("Set3"))
axes[2].set_title('Application Types')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{PROC}/fig_interventions.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved figure to {PROC}/fig_interventions.png")

# %%
# =============================================================
# 5. GROUP DEMOGRAPHICS — Distribution plots
# =============================================================
print("\n=== Group Demographics ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Age distribution
df_groups_wide['age'].dropna().hist(bins=30, ax=axes[0, 0], color='steelblue', edgecolor='white')
axes[0, 0].set_title('Age Distribution (Group Means)')
axes[0, 0].set_xlabel('Age (years)')

# Weight distribution
df_groups_wide['weight'].dropna().hist(bins=30, ax=axes[0, 1], color='coral', edgecolor='white')
axes[0, 1].set_title('Weight Distribution (Group Means)')
axes[0, 1].set_xlabel('Weight (kg)')

# BMI distribution
if 'bmi' in df_groups_wide.columns:
    df_groups_wide['bmi'].dropna().hist(bins=25, ax=axes[0, 2], color='mediumpurple', edgecolor='white')
    axes[0, 2].set_title('BMI Distribution (Group Means)')
    axes[0, 2].set_xlabel('BMI')

# Sex distribution
if 'sex' in df_groups_wide.columns:
    sex_counts = df_groups_wide['sex'].value_counts()
    sex_counts.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral', 'gray'])
    axes[1, 0].set_title('Sex Distribution')
    axes[1, 0].tick_params(axis='x', rotation=0)

# Smoking status
if 'smoking' in df_groups_wide.columns:
    smoke_counts = df_groups_wide['smoking'].value_counts()
    smoke_counts.plot(kind='bar', ax=axes[1, 1], color=sns.color_palette("Set2"))
    axes[1, 1].set_title('Smoking Status')
    axes[1, 1].tick_params(axis='x', rotation=45)

# Species
if 'species' in df_groups_wide.columns:
    species_counts = df_groups_wide['species'].value_counts().head(5)
    species_counts.plot(kind='bar', ax=axes[1, 2], color=sns.color_palette("Paired"))
    axes[1, 2].set_title('Species')
    axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{PROC}/fig_group_demographics.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved figure to {PROC}/fig_group_demographics.png")

# %%
# =============================================================
# 6. INDIVIDUAL DEMOGRAPHICS — Distribution plots
# =============================================================
print("\n=== Individual Demographics ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

df_indiv_wide['age'].dropna().hist(bins=40, ax=axes[0, 0], color='steelblue', edgecolor='white')
axes[0, 0].set_title(f"Age (n={df_indiv_wide['age'].notna().sum():,})")
axes[0, 0].set_xlabel('Age (years)')

df_indiv_wide['weight'].dropna().hist(bins=40, ax=axes[0, 1], color='coral', edgecolor='white')
axes[0, 1].set_title(f"Weight (n={df_indiv_wide['weight'].notna().sum():,})")
axes[0, 1].set_xlabel('Weight (kg)')

if 'bmi' in df_indiv_wide.columns:
    df_indiv_wide['bmi'].dropna().hist(bins=30, ax=axes[0, 2], color='mediumpurple', edgecolor='white')
    axes[0, 2].set_title(f"BMI (n={df_indiv_wide['bmi'].notna().sum():,})")

if 'sex' in df_indiv_wide.columns:
    df_indiv_wide['sex'].value_counts().plot(kind='bar', ax=axes[1, 0],
                                              color=['steelblue', 'coral', 'gray'])
    axes[1, 0].set_title('Sex')
    axes[1, 0].tick_params(axis='x', rotation=0)

if 'cyp2d6 phenotype' in df_indiv_wide.columns:
    df_indiv_wide['cyp2d6 phenotype'].value_counts().head(6).plot(
        kind='bar', ax=axes[1, 1], color=sns.color_palette("Set2"))
    axes[1, 1].set_title('CYP2D6 Phenotype')
    axes[1, 1].tick_params(axis='x', rotation=45)

if 'ethnicity' in df_indiv_wide.columns:
    df_indiv_wide['ethnicity'].value_counts().head(6).plot(
        kind='bar', ax=axes[1, 2], color=sns.color_palette("Paired"))
    axes[1, 2].set_title('Ethnicity')
    axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{PROC}/fig_individual_demographics.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved figure to {PROC}/fig_individual_demographics.png")

# %%
# =============================================================
# 7. CAFFEINE STUDY ANALYSIS — Our primary focus drug
# =============================================================
print("\n=== Caffeine Study Analysis ===")

print(f"Total caffeine studies: {len(df_caff)}")
print(f"Total outputs referenced: {df_caff['output_count'].sum():,}")
print(f"Total individuals: {df_caff['individual_count'].sum():,}")
print(f"Total groups: {df_caff['group_count'].sum():,}")

print(f"\nLicence distribution:")
print(df_caff['licence'].value_counts())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Outputs per study
df_caff_sorted = df_caff.sort_values('output_count', ascending=False).head(20)
axes[0].barh(df_caff_sorted['name'], df_caff_sorted['output_count'], color='steelblue')
axes[0].set_title('Top 20 Caffeine Studies by Output Count')
axes[0].set_xlabel('Output Count')
axes[0].invert_yaxis()

# Distribution of outputs per study
df_caff['output_count'].hist(bins=30, ax=axes[1], color='coral', edgecolor='white')
axes[1].set_title('Distribution of Outputs per Caffeine Study')
axes[1].set_xlabel('Output Count')
axes[1].set_ylabel('Number of Studies')

# Individuals per study
df_caff['individual_count'].hist(bins=30, ax=axes[2], color='mediumpurple', edgecolor='white')
axes[2].set_title('Distribution of Individuals per Caffeine Study')
axes[2].set_xlabel('Individual Count')
axes[2].set_ylabel('Number of Studies')

plt.tight_layout()
plt.savefig(f"{PROC}/fig_caffeine_studies.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# =============================================================
# 8. SUBSTANCE LANDSCAPE — Top drugs in PK-DB
# =============================================================
print("\n=== Substance Landscape ===")

fig, ax = plt.subplots(figsize=(12, 8))
top30 = df_subs.head(30)
colors = ['#e74c3c' if 'caffeine' in str(x).lower() else '#3498db'
          for x in top30['info_node__label']]
ax.barh(top30['info_node__label'], top30['output_count'], color=colors)
ax.set_title('Top 30 Substances by PK Output Count', fontsize=14)
ax.set_xlabel('Output Count')
ax.invert_yaxis()

# Highlight caffeine
ax.annotate('PRIMARY TARGET', xy=(9396, 0), fontsize=9, fontweight='bold',
            color='#e74c3c', va='center', ha='left')

plt.tight_layout()
plt.savefig(f"{PROC}/fig_substance_landscape.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# =============================================================
# 9. CORRELATION MATRIX — Numeric covariates
# =============================================================
print("\n=== Covariate Correlations (Groups) ===")

numeric_cols = ['age', 'weight', 'bmi', 'height', 'group_count']
available = [c for c in numeric_cols if c in df_groups_wide.columns]
corr_data = df_groups_wide[available].dropna(how='all')

if len(corr_data) > 10:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = corr_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Covariate Correlation Matrix (Group-level)')
    plt.tight_layout()
    plt.savefig(f"{PROC}/fig_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()

# %%
# =============================================================
# 10. CAFFEINE-SPECIFIC GROUPS — Demographics of caffeine cohorts
# =============================================================
print("\n=== Caffeine-Specific Group Demographics ===")

# Get caffeine study SIDs
caff_sids = set(df_caff['sid'].tolist())
df_caff_groups = df_groups_wide[df_groups_wide['study_sid'].isin(caff_sids)].copy()
print(f"Caffeine groups: {len(df_caff_groups):,}")

if len(df_caff_groups) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age vs Weight scatter
    ax = axes[0, 0]
    mask = df_caff_groups[['age', 'weight']].notna().all(axis=1)
    if mask.sum() > 5:
        ax.scatter(df_caff_groups.loc[mask, 'age'],
                   df_caff_groups.loc[mask, 'weight'],
                   alpha=0.5, s=df_caff_groups.loc[mask, 'group_count'] * 3,
                   c='steelblue', edgecolors='white')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Weight (kg)')
        ax.set_title('Caffeine Groups: Age vs Weight\n(bubble size = group count)')

    # Smoking status in caffeine studies
    ax = axes[0, 1]
    if 'smoking' in df_caff_groups.columns:
        smoke = df_caff_groups['smoking'].value_counts()
        smoke.plot(kind='bar', ax=ax, color=sns.color_palette("Set2"))
        ax.set_title('Smoking Status in Caffeine Studies')
        ax.tick_params(axis='x', rotation=45)

    # Sex in caffeine studies
    ax = axes[1, 0]
    if 'sex' in df_caff_groups.columns:
        sex = df_caff_groups['sex'].value_counts()
        sex.plot(kind='bar', ax=ax, color=['steelblue', 'coral', 'gray'])
        ax.set_title('Sex Distribution in Caffeine Studies')
        ax.tick_params(axis='x', rotation=0)

    # Healthy status
    ax = axes[1, 1]
    if 'healthy' in df_caff_groups.columns:
        health = df_caff_groups['healthy'].value_counts()
        health.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#95a5a6'])
        ax.set_title('Health Status in Caffeine Studies')
        ax.tick_params(axis='x', rotation=0)

    plt.suptitle('Caffeine Study Group Demographics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PROC}/fig_caffeine_demographics.png", dpi=150, bbox_inches='tight')
    plt.show()

# %%
# =============================================================
# 11. DATA QUALITY SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("  DATA QUALITY SUMMARY")
print("=" * 60)

print(f"\n  Groups (wide format):")
print(f"    Total: {len(df_groups_wide):,}")
print(f"    With age: {df_groups_wide['age'].notna().sum():,}")
print(f"    With weight: {df_groups_wide['weight'].notna().sum():,}")
if 'sex' in df_groups_wide.columns:
    print(f"    With sex: {df_groups_wide['sex'].notna().sum():,}")
if 'smoking' in df_groups_wide.columns:
    print(f"    With smoking: {df_groups_wide['smoking'].notna().sum():,}")

print(f"\n  Individuals (wide format):")
print(f"    Total: {len(df_indiv_wide):,}")
print(f"    With age: {df_indiv_wide['age'].notna().sum():,}")
print(f"    With weight: {df_indiv_wide['weight'].notna().sum():,}")
if 'sex' in df_indiv_wide.columns:
    print(f"    With sex: {df_indiv_wide['sex'].notna().sum():,}")

print(f"\n  Interventions (dosing):")
print(f"    Total: {len(df_doses):,}")
print(f"    Substances: {df_doses['substance'].nunique()}")
print(f"    Routes: {df_doses['route'].nunique()}")

print(f"\n  Caffeine focus:")
print(f"    Studies: {len(df_caff)}")
print(f"    Total referenced outputs: {df_caff['output_count'].sum():,}")
caff_grp_count = len(df_caff_groups) if 'df_caff_groups' in dir() else 0
print(f"    Groups with demographics: {caff_grp_count}")

print(f"\n  Processed files saved to {PROC}/:")
for f in sorted(os.listdir(PROC)):
    size = os.path.getsize(os.path.join(PROC, f))
    print(f"    {f} ({size/1024:.1f} KB)")

print(f"\n{'='*60}")
print("  EDA COMPLETE!")
print(f"{'='*60}")
print("""
Next steps:
  1. Build PK feature engineering (BSA, creatinine clearance)
  2. Start ML modeling on TDC clearance data (once downloaded)
  3. Connect demographics to caffeine clearance outcomes
  4. Power BI dashboard from processed CSVs
""")