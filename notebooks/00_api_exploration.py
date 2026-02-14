"""
PK-DB API Exploration — v2 (based on actual API structure)
==========================================================
The API uses nested pagination: {current_page, last_page, data: {count, data: [...]}}
Data must be pulled per-study (outputs/timecourses require study context)
"""

# %%
import requests
import pandas as pd
import json
from time import sleep

BASE_URL = "https://pk-db.com/api/v1"

def safe_get(url):
    """Fetch JSON with error handling."""
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def get_records(url):
    """Extract records from PK-DB's nested pagination format.
    Format: {current_page, last_page, data: {count, data: [...]}}
    """
    raw = safe_get(url)
    if isinstance(raw, dict):
        # Nested format: data.data
        inner = raw.get("data", raw)
        if isinstance(inner, dict) and "data" in inner:
            return inner["data"], inner.get("count", 0), raw
        elif isinstance(inner, list):
            return inner, len(inner), raw
        # Single object (like info_nodes)
        return [raw], 1, raw
    elif isinstance(raw, list):
        return raw, len(raw), raw
    return [], 0, raw

# %%
# =============================================================
# 1. DATABASE STATS
# =============================================================
stats = safe_get(f"{BASE_URL}/statistics/?format=json")
print("=== PK-DB v{} ===".format(stats['version']))
for k, v in stats.items():
    if k != 'version':
        print(f"  {k}: {v:,}")

# %%
# =============================================================
# 2. GET STUDY LIST — see all available studies
# =============================================================
records, count, raw = get_records(f"{BASE_URL}/studies/?format=json&limit=50")
print(f"Studies returned: {len(records)} (total: {count})")
print(f"Pages: {raw.get('last_page', '?')}")

print(f"\n--- Study fields ({len(records[0])} fields) ---")
for k, v in records[0].items():
    print(f"  {k} ({type(v).__name__}): {str(v)[:100]}")

print(f"\n--- First 50 study names ---")
for i, s in enumerate(records):
    print(f"  {i+1}. {s['sid']} — {s['name']} "
          f"(outputs:{s.get('output_count',0)}, "
          f"individuals:{s.get('individual_count',0)})")

# %%
# =============================================================
# 3. PICK A DATA-RICH STUDY AND EXPLORE ITS STRUCTURE
#    Let's use the study UUID to pull its outputs
# =============================================================
# Pick first study that has substantial outputs
rich_study = None
for s in records:
    if s.get('output_count', 0) > 50:
        rich_study = s
        break

if not rich_study:
    rich_study = records[0]

print(f"Selected study: {rich_study['sid']} — {rich_study['name']}")
print(f"  Outputs: {rich_study.get('output_count', 0)}")
print(f"  Individuals: {rich_study.get('individual_count', 0)}")
print(f"  Groups: {rich_study.get('group_count', 0)}")

study_sid = rich_study['sid']

# %%
# =============================================================
# 4. EXPLORE OUTPUTS FOR THIS SPECIFIC STUDY
# =============================================================
print(f"\n=== OUTPUTS for {study_sid} ===")

# Try with study filter
outputs_urls = [
    f"{BASE_URL}/outputs/?format=json&limit=5&study_sid={study_sid}",
    f"{BASE_URL}/outputs/?format=json&limit=5&study_name={rich_study['name']}",
    f"{BASE_URL}/outputs/?format=json&limit=5&study={study_sid}",
    f"{BASE_URL}/outputs/{study_sid}/?format=json&limit=5",
]

for url in outputs_urls:
    sleep(0.5)
    try:
        recs, cnt, _ = get_records(url)
        print(f"\n  URL: {url}")
        print(f"  Count: {cnt}, Records: {len(recs)}")
        if recs and isinstance(recs[0], dict):
            print(f"  Fields: {list(recs[0].keys())[:15]}")
            print(f"  Sample: {json.dumps(recs[0], indent=2)[:1500]}")
            break  # found working URL
    except Exception as e:
        print(f"  URL: {url}")
        print(f"  Error: {e}")

# %%
# =============================================================
# 5. TRY STUDY-SPECIFIC DOWNLOAD ENDPOINT
#    The paper mentions downloading full study data as ZIP
# =============================================================
print(f"\n=== STUDY DOWNLOAD ENDPOINTS ===")

download_urls = [
    f"{BASE_URL}/studies/{study_sid}/?format=json",
    f"{BASE_URL}/studies/{rich_study['pk']}/?format=json",
    f"{BASE_URL}/studies/?format=json&sid={study_sid}",
]

for url in download_urls:
    sleep(0.5)
    try:
        data = safe_get(url)
        print(f"\n  URL: {url}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:20]}")
            print(f"  Preview: {json.dumps(data, indent=2)[:2000]}")
        break
    except Exception as e:
        print(f"  URL: {url}")
        print(f"  Error: {e}")

# %%
# =============================================================
# 6. DEEP DIVE INTO INDIVIDUALS — Full field structure
# =============================================================
print(f"\n=== INDIVIDUALS (full field structure) ===")
recs, cnt, _ = get_records(f"{BASE_URL}/individuals/?format=json&limit=3")
print(f"Total individuals: {cnt}")

if recs:
    print(f"\n--- FULL first individual record ---")
    print(json.dumps(recs[0], indent=2))

# %%
# =============================================================
# 7. DEEP DIVE INTO INTERVENTIONS — Full field structure
# =============================================================
print(f"\n=== INTERVENTIONS (full field structure) ===")
recs, cnt, _ = get_records(f"{BASE_URL}/interventions/?format=json&limit=3")
print(f"Total interventions: {cnt}")

if recs:
    print(f"\n--- FULL first intervention record ---")
    print(json.dumps(recs[0], indent=2))

# %%
# =============================================================
# 8. DEEP DIVE INTO GROUPS — Full field structure
# =============================================================
print(f"\n=== GROUPS (full field structure) ===")
recs, cnt, _ = get_records(f"{BASE_URL}/groups/?format=json&limit=3")
print(f"Total groups: {cnt}")

if recs:
    print(f"\n--- FULL first group record ---")
    print(json.dumps(recs[0], indent=2))

# %%
# =============================================================
# 9. TRY FETCHING STUDY DATA VIA UUID
#    The API notebook mentions: /individuals/?uuid=<study_uuid>
# =============================================================
print(f"\n=== UUID-BASED FETCHING ===")

# Check if studies have a uuid field
study_uuid = rich_study.get('uuid', rich_study.get('pk', None))
print(f"Study pk: {rich_study.get('pk')}")
print(f"Study uuid: {rich_study.get('uuid', 'NOT FOUND')}")
print(f"Study sid: {rich_study.get('sid')}")

# Try uuid-based endpoints from the API notebook
if study_uuid:
    uuid_urls = [
        f"{BASE_URL}/individuals/?uuid={study_uuid}&format=json&limit=3",
        f"{BASE_URL}/interventions/?uuid={study_uuid}&format=json&limit=3",
        f"{BASE_URL}/outputs/?uuid={study_uuid}&format=json&limit=3",
    ]
    for url in uuid_urls:
        sleep(0.5)
        try:
            recs, cnt, _ = get_records(url)
            print(f"\n  URL: ...{url.split('?')[1]}")
            print(f"  Count: {cnt}, Records: {len(recs)}")
        except Exception as e:
            print(f"\n  URL: ...{url.split('?')[1]}")
            print(f"  Error: {e}")

# %%
# =============================================================
# 10. CHECK ALL API ENDPOINTS AVAILABLE
#     Hit the API root to see what's available
# =============================================================
print(f"\n=== API ROOT — All available endpoints ===")
try:
    root = safe_get(f"{BASE_URL}/?format=json")
    print(json.dumps(root, indent=2)[:3000])
except Exception as e:
    print(f"Error: {e}")

# %%
print("""
=============================================================
  COPY ALL OUTPUT ABOVE AND PASTE IT BACK TO ME!
  
  I especially need:
  1. The working URL pattern for OUTPUTS
  2. Full field structure of individuals/interventions/groups
  3. The API root endpoint list
  4. Whether UUID-based fetching works
  
  This determines our entire pipeline architecture.
=============================================================
""")