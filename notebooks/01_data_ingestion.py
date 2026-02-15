"""
PK-DB + TDC Data Ingestion
===========================
Multi-source ingestion:
  Source 1: PK-DB API (pkdata/) — demographics, interventions, study metadata
  Source 2: TDC (Harvard Dataverse) — ADME/PK endpoint datasets (clearance, half-life, etc.)

This notebook pulls all available data and saves to data/raw/
"""

# %%
import requests
import pandas as pd
import json
import os
import zipfile
import io
from time import sleep

BASE_URL = "https://pk-db.com/api/v1"
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def get_all_pages(endpoint, max_pages=None, delay=0.3):
    """Paginate through a pkdata/ endpoint and collect all records.
    API returns 20 records/page regardless of limit param.
    """
    all_records = []
    page = 1

    while True:
        url = f"{BASE_URL}/{endpoint}/?format=json&page={page}"
        resp = requests.get(url)
        resp.raise_for_status()
        raw = resp.json()

        inner = raw.get("data", {})
        if isinstance(inner, dict):
            records = inner.get("data", [])
            total = inner.get("count", 0)
        else:
            records = inner if isinstance(inner, list) else []
            total = len(records)

        all_records.extend(records)
        last_page = raw.get("last_page", 1)

        if page == 1 or page % 100 == 0 or page == last_page:
            pct = len(all_records) / total * 100 if total > 0 else 0
            print(f"  Page {page}/{last_page} — {len(all_records):,}/{total:,} ({pct:.1f}%)")

        if page >= last_page:
            break
        if max_pages and page >= max_pages:
            print(f"  Stopped at max_pages={max_pages} — got {len(all_records):,} records")
            break

        page += 1
        sleep(delay)

    return all_records, total

# %%
# =============================================================
# 1. PULL ALL PKDATA/STUDIES
# =============================================================
print("=== 1. Pulling pkdata/studies ===")
studies, total = get_all_pages("pkdata/studies")
df_studies = pd.DataFrame(studies)
df_studies.to_csv(f"{RAW_DIR}/pkdb_studies.csv", index=False)
print(f"Saved {len(df_studies)} studies")
print(f"Columns: {list(df_studies.columns)}")

# %%
# =============================================================
# 2. PULL ALL PKDATA/GROUPS (demographics at cohort level)
# =============================================================
print("\n=== 2. Pulling pkdata/groups ===")
groups, total = get_all_pages("pkdata/groups")
df_groups = pd.DataFrame(groups)
df_groups.to_csv(f"{RAW_DIR}/pkdb_groups.csv", index=False)
print(f"Saved {len(df_groups):,} group records")
print(f"\nMeasurement types:")
print(df_groups['measurement_type'].value_counts().head(15))

# %%
# =============================================================
# 3. PULL PKDATA/INDIVIDUALS (demographics at patient level)
#    API returns 20/page → 159K = ~8000 pages = too slow
#    Cap at 2000 pages (~40K records) — enough for EDA + modeling
#    Full pull → Sachi's async batch client later
# =============================================================
print("\n=== 3. Pulling pkdata/individuals (capped at 2000 pages) ===")
print("(~40K records, ~10 min with 0.3s delay)")
individuals, total = get_all_pages("pkdata/individuals", max_pages=2000, delay=0.3)
df_individuals = pd.DataFrame(individuals)
df_individuals.to_csv(f"{RAW_DIR}/pkdb_individuals.csv", index=False)
print(f"\nSaved {len(df_individuals):,} of {total:,} individual records")
print(f"\nMeasurement types:")
print(df_individuals['measurement_type'].value_counts().head(15))

# %%
# =============================================================
# 4. PULL ALL PKDATA/INTERVENTIONS (dosing info)
# =============================================================
print("\n=== 4. Pulling pkdata/interventions ===")
interventions, total = get_all_pages("pkdata/interventions")
df_interventions = pd.DataFrame(interventions)
df_interventions.to_csv(f"{RAW_DIR}/pkdb_interventions.csv", index=False)
print(f"Saved {len(df_interventions):,} interventions")
print(f"\nTop substances:")
print(df_interventions['substance'].value_counts().head(15))
print(f"\nRoutes:")
print(df_interventions['route'].value_counts())

# %%
# =============================================================
# 5. PULL SUBSTANCE STATISTICS
# =============================================================
print("\n=== 5. Pulling substance statistics ===")
resp = requests.get(f"{BASE_URL}/statistics/substances/?format=json")
sub_stats = resp.json()
df_substances = pd.DataFrame(sub_stats)
df_substances = df_substances.sort_values('output_count', ascending=False)
df_substances.to_csv(f"{RAW_DIR}/pkdb_substance_stats.csv", index=False)
print(f"Saved {len(df_substances)} substances")
print(f"\nTop 15 by output count:")
print(df_substances.head(15).to_string(index=False))

# %%
# =============================================================
# 6. PULL STUDY DETAILS FOR CAFFEINE-RELATED STUDIES
# =============================================================
print("\n=== 6. Pulling caffeine study details ===")

caffeine_studies = df_studies[
    df_studies['substances'].apply(
        lambda x: 'caffeine' in str(x).lower() if pd.notna(x) else False
    )
]
print(f"Studies with caffeine: {len(caffeine_studies)}")

study_details = []
for _, row in caffeine_studies.iterrows():
    sid = row['sid']
    sleep(0.5)
    try:
        detail = requests.get(f"{BASE_URL}/studies/{sid}/?format=json").json()
        study_details.append({
            'sid': sid,
            'name': detail.get('name'),
            'output_count': detail.get('output_count', 0),
            'individual_count': detail.get('individual_count', 0),
            'group_count': detail.get('group_count', 0),
            'intervention_count': detail.get('intervention_count', 0),
            'timecourse_count': detail.get('timecourse_count', 0),
            'output_pks': detail.get('outputset', {}).get('outputs', []),
            'licence': detail.get('licence'),
        })
        print(f"  {sid} — {detail.get('name')} | outputs={detail.get('output_count',0)}")
    except Exception as e:
        print(f"  {sid} — Error: {e}")

df_caffeine_details = pd.DataFrame(study_details)
df_caffeine_details.to_csv(f"{RAW_DIR}/pkdb_caffeine_study_details.csv", index=False)
print(f"\nSaved {len(df_caffeine_details)} caffeine study details")

# %%
# =============================================================
# 7. PULL TDC ADME DATASETS — Direct download (no PyTDC needed)
#    TDC data is hosted on Harvard Dataverse
# =============================================================
print("\n=== 7. Pulling TDC ADME Datasets (direct download) ===")

TDC_BASE = "https://dataverse.harvard.edu/api/access/datafile"

tdc_downloads = [
    ("Clearance_AstraZeneca", "6358080"),
    ("Half_Life_Obach", "6358082"),
    ("Bioavailability_Ma", "6358078"),
    ("PPBR_AZ", "6358084"),
]

for name, file_id in tdc_downloads:
    print(f"\n--- {name} ---")
    url = f"{TDC_BASE}/{file_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content

        # Detect format: zip or flat CSV/TSV
        if content[:4] == b'\x50\x4b\x03\x04':
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                fname = [f for f in zf.namelist()
                         if f.endswith(('.csv', '.tsv', '.tab', '.txt'))]
                target = fname[0] if fname else zf.namelist()[0]
                with zf.open(target) as f:
                    df = pd.read_csv(f, sep=None, engine='python')
        else:
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep='\t')

        outpath = f"{RAW_DIR}/tdc_{name.lower()}.csv"
        df.to_csv(outpath, index=False)
        print(f"  Records: {len(df)} | Columns: {list(df.columns)}")
        print(f"  Saved to: {outpath}")
        print(df.head(2))

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Fallback: https://tdcommons.ai/single_pred_tasks/adme/")

# %%
# =============================================================
# 8. DATA INVENTORY — Summary of everything we pulled
# =============================================================
print("\n" + "=" * 60)
print("  DATA INVENTORY — Raw files in data/raw/")
print("=" * 60)

for f in sorted(os.listdir(RAW_DIR)):
    if f.endswith('.csv'):
        path = os.path.join(RAW_DIR, f)
        df_temp = pd.read_csv(path, nrows=0)
        size = os.path.getsize(path)
        nrows = sum(1 for _ in open(path, encoding='utf-8', errors='ignore')) - 1
        print(f"\n  {f}")
        print(f"    Rows: {nrows:,} | Size: {size/1024:.1f} KB")
        print(f"    Columns: {list(df_temp.columns)}")

print(f"\n{'='*60}")
print("  INGESTION COMPLETE!")
print(f"{'='*60}")
print("""
Next steps (Vaibhav):
  1. Write transformation/cleaning logic
  2. Build EDA notebook
  3. Define PK feature engineering functions

Hand off to Sachi:
  - Schema file: config/schema.sql (already done)
  - Pagination: 20 records/page, use page param
  - Full individuals pull needs async/threaded client
  - Rate limit: 0.3s between requests minimum
""")