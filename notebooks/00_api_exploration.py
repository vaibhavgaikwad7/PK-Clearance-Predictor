"""
PK-DB API Exploration — v5 (FINAL TARGETED PROBE)
===================================================
We know:
  - pkdata/ works for studies, groups, individuals, interventions
  - pkdata/outputs returns 0 (likely requires auth or different access)
  - Study detail has outputset with output PKs [160175, 160176, ...]
  - /outputs/ endpoint exists but needs correct query

This script: 
  1. Tries to access individual output records by PK
  2. Tests the regular /outputs/ endpoint with study-based filters  
  3. Pulls larger samples from working pkdata/ endpoints
  4. Gets full substance stats for drug selection
"""

# %%
import requests
import pandas as pd
import json
from time import sleep

BASE_URL = "https://pk-db.com/api/v1"

def safe_get(url):
    print(f"  GET {url[:130]}...")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def get_records(raw):
    if isinstance(raw, dict):
        inner = raw.get("data", {})
        if isinstance(inner, dict) and "data" in inner:
            return inner["data"], inner.get("count", 0)
        if isinstance(inner, list):
            return inner, len(inner)
    if isinstance(raw, list):
        return raw, len(raw)
    return [], 0

# %%
# =============================================================
# 1. ACCESS INDIVIDUAL OUTPUT RECORDS BY PK
#    outputset gave us PKs like 160175 — try fetching directly
# =============================================================
print("=== FETCHING OUTPUT RECORDS BY PK ===\n")

output_pks = [160175, 160176, 160177]  # from PKDB00954's outputset

for pk in output_pks:
    urls = [
        f"{BASE_URL}/outputs/{pk}/?format=json",
        f"{BASE_URL}/outputs/?pk={pk}&format=json",
        f"{BASE_URL}/outputs/?format=json&pk={pk}",
    ]
    for url in urls:
        sleep(0.3)
        try:
            data = safe_get(url)
            recs, cnt = get_records(data)
            if cnt > 0 or (isinstance(data, dict) and 'pk' in data):
                print(f"  ✅ FOUND! URL pattern: {url}")
                result = data if 'pk' in data else (recs[0] if recs else data)
                print(json.dumps(result, indent=2)[:3000])
                break
            else:
                print(f"  count={cnt}")
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP {e.response.status_code}")
        except Exception as e:
            print(f"  Error: {str(e)[:80]}")
    else:
        continue
    break  # stop after first successful pattern

# %%
# =============================================================
# 2. TRY /outputs/ WITH STUDY SID IN URL PATH
#    Maybe: /outputs/?study_sid=PKDB00954 (relational endpoint)
# =============================================================
print("\n=== OUTPUTS WITH DIFFERENT FILTER COMBOS ===\n")

# Get a study with open licence (closed might restrict outputs)
raw = safe_get(f"{BASE_URL}/pkdata/studies/?format=json&limit=50")
recs, _ = get_records(raw)

# Find open-licence studies
open_studies = [s for s in recs if s.get('licence') == 'open']
print(f"Open-licence studies in first 50: {len(open_studies)}")
for s in open_studies[:5]:
    print(f"  {s['sid']} — {s['name']} | substances: {s.get('substances', [])}")

if open_studies:
    os = open_studies[0]
    study_sid = os['sid']
    study_name = os['name']
    print(f"\nTrying outputs for OPEN study: {study_sid} ({study_name})")
    
    test_urls = [
        f"{BASE_URL}/outputs/?format=json&limit=5&study_sid={study_sid}",
        f"{BASE_URL}/outputs/?format=json&limit=5&study__sid={study_sid}",
        f"{BASE_URL}/pkdata/outputs/?format=json&limit=5&study_sid={study_sid}",
        f"{BASE_URL}/pkdata/outputs/?format=json&limit=5&study_name={study_name}",
        f"{BASE_URL}/pkdata/outputs/?format=json&study={study_sid}",
    ]
    
    for url in test_urls:
        sleep(0.3)
        try:
            data = safe_get(url)
            recs, cnt = get_records(data)
            print(f"  count={cnt}, records={len(recs)}")
            if cnt > 0:
                print(f"  ✅ FOUND!")
                print(f"  Fields: {list(recs[0].keys())}")
                print(json.dumps(recs[0], indent=2)[:2000])
                break
        except Exception as e:
            print(f"  Error: {str(e)[:80]}")

# %%
# =============================================================
# 3. CHECK IF LICENCE MATTERS — Compare open vs closed study detail
# =============================================================
print("\n=== OPEN vs CLOSED STUDY — outputset comparison ===\n")

if open_studies:
    open_sid = open_studies[0]['sid']
    print(f"Open study: {open_sid}")
    try:
        detail = safe_get(f"{BASE_URL}/studies/{open_sid}/?format=json")
        oset = detail.get('outputset', {})
        outputs_list = oset.get('outputs', [])
        print(f"  outputset.outputs count: {len(outputs_list)}")
        if outputs_list:
            print(f"  First 5 output PKs: {outputs_list[:5]}")
            
            # Now try fetching one of these output PKs
            opk = outputs_list[0]
            print(f"\n  Fetching output pk={opk}...")
            try:
                out_data = safe_get(f"{BASE_URL}/outputs/{opk}/?format=json")
                print(f"  ✅ Got output record!")
                print(json.dumps(out_data, indent=2)[:3000])
            except requests.exceptions.HTTPError as e:
                print(f"  HTTP {e.response.status_code}")
                
                # Try the interventionset/individualset patterns
                iset = detail.get('interventionset', {})
                print(f"\n  interventionset keys: {list(iset.keys()) if isinstance(iset, dict) else type(iset)}")
                
                dset = detail.get('dataset', {})
                print(f"  dataset keys: {list(dset.keys()) if isinstance(dset, dict) else type(dset)}")
                if isinstance(dset, dict):
                    print(f"  dataset: {json.dumps(dset, indent=2)[:1500]}")
    except Exception as e:
        print(f"  Error: {e}")

# %%
# =============================================================
# 4. BULK PULL LARGER SAMPLES FROM WORKING ENDPOINTS
#    Get 50 records each to understand data distribution
# =============================================================
print("\n=== LARGER SAMPLE: pkdata/individuals (50 records) ===\n")

raw = safe_get(f"{BASE_URL}/pkdata/individuals/?format=json&limit=50")
recs, cnt = get_records(raw)
df_indiv = pd.DataFrame(recs)
print(f"Total: {cnt} | Sample: {len(df_indiv)}")
print(f"\nMeasurement types in sample:")
print(df_indiv['measurement_type'].value_counts().to_string())
print(f"\nStudies in sample:")
print(df_indiv['study_name'].value_counts().to_string())

# %%
print("\n=== LARGER SAMPLE: pkdata/groups (50 records) ===\n")

raw = safe_get(f"{BASE_URL}/pkdata/groups/?format=json&limit=50")
recs, cnt = get_records(raw)
df_groups = pd.DataFrame(recs)
print(f"Total: {cnt} | Sample: {len(df_groups)}")
print(f"\nMeasurement types in sample:")
print(df_groups['measurement_type'].value_counts().to_string())

# %%
print("\n=== LARGER SAMPLE: pkdata/interventions (50 records) ===\n")

raw = safe_get(f"{BASE_URL}/pkdata/interventions/?format=json&limit=50")
recs, cnt = get_records(raw)
df_interv = pd.DataFrame(recs)
print(f"Total: {cnt} | Sample: {len(df_interv)}")
print(f"\nSubstances in sample:")
print(df_interv['substance'].value_counts().head(15).to_string())
print(f"\nRoutes in sample:")
print(df_interv['route'].value_counts().to_string())
print(f"\nMeasurement types:")
print(df_interv['measurement_type'].value_counts().to_string())

# %%
# =============================================================
# 5. FULL SUBSTANCE STATISTICS — which drugs have most data
# =============================================================
print("\n=== FULL SUBSTANCE STATISTICS ===\n")

sub_stats = safe_get(f"{BASE_URL}/statistics/substances/?format=json")

if isinstance(sub_stats, list):
    df_subs = pd.DataFrame(sub_stats)
    df_subs = df_subs.sort_values('output_count', ascending=False)
    print(f"Total substances: {len(df_subs)}")
    print(f"\n--- TOP 30 SUBSTANCES BY OUTPUT COUNT ---")
    print(df_subs.head(30).to_string(index=False))
elif isinstance(sub_stats, dict):
    print(json.dumps(sub_stats, indent=2)[:3000])

# %%
print("""
=============================================================
  🎯 FINAL OUTPUT NEEDED:
  
  1. Did fetching output by PK work? (Step 1 & 3)
  2. Open vs closed study — does licence affect access?
  3. Measurement types in individuals & groups (Steps 4)
  4. TOP 30 substances by output count (Step 5)
  
  After this we BUILD — no more exploration!
=============================================================
""")
# %%
