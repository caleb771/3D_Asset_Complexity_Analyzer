import requests
import os
import zipfile
import datetime
import time
import random
import sys
import re

API_TOKEN = "0c0098a832675396f037ffd05044753b"

# ── TIER TARGETS ──────────────────────────────────────────────────────────────
TIERS = {
    "extreme": {"face_min": 1_000_000, "face_max": 999_999_999, "target": 180},
}
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\USER PC\Downloads\demo\demo_new"
os.makedirs(BASE_DIR, exist_ok=True)

SEARCH_URL = "https://api.sketchfab.com/v3/models"
headers    = {"Authorization": f"Token {API_TOKEN}"}

# Per-tier state
state = {}
for tier, cfg in TIERS.items():
    folder = os.path.join(BASE_DIR, tier)
    os.makedirs(folder, exist_ok=True)

    completed_path = os.path.join(folder, "completed.txt")
    cursor_path    = os.path.join(folder, "cursor.txt")

    completed = set()
    if os.path.exists(completed_path):
        with open(completed_path, "r") as f:
            completed = set(line.strip() for line in f)

    cursor = None
    if os.path.exists(cursor_path):
        with open(cursor_path, "r") as f:
            cursor = f.read().strip() or None

    state[tier] = {
        "folder":         folder,
        "completed_path": completed_path,
        "cursor_path":    cursor_path,
        "completed":      completed,
        "cursor":         cursor,
        "downloaded":     len(completed),
        "target":         cfg["target"],
        "face_min":       cfg["face_min"],
        "face_max":       cfg["face_max"],
    }

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)

def safe_get(url, headers, params=None):
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        return resp
    elif resp.status_code == 429:
        log("Rate limit hit — saving cursors and exiting.")
        for tier, s in state.items():
            with open(s["cursor_path"], "w") as f:
                f.write(s["cursor"] or "")
        sys.exit(0)
    else:
        log(f"Error {resp.status_code} on {url}")
        resp.raise_for_status()

def all_done():
    return all(s["downloaded"] >= s["target"] for s in state.values())

def active_tiers():
    return {t: s for t, s in state.items() if s["downloaded"] < s["target"]}

# Use a shared cursor so one page fetch serves both tiers simultaneously
shared_cursor = None
shared_cursor_path = os.path.join(BASE_DIR, "shared_cursor.txt")
if os.path.exists(shared_cursor_path):
    with open(shared_cursor_path, "r") as f:
        shared_cursor = f.read().strip() or None

log("=" * 60)
for tier, s in state.items():
    log(f"  {tier.upper():8s} — target: {s['target']} | already downloaded: {s['downloaded']} | remaining: {s['target'] - s['downloaded']}")
log("=" * 60)

while not all_done():
    params = {
        "downloadable": "true",
        "sort_by":      "publishedAt",
        "sort_order":   "desc",
        "count":        100,
    }
    if shared_cursor:
        params["cursor"] = shared_cursor

    resp    = safe_get(SEARCH_URL, headers, params=params)
    data    = resp.json()
    results = data.get("results", [])
    shared_cursor = data.get("cursors", {}).get("next")

    if not results or not shared_cursor:
        log("No more results or cursor exhausted.")
        break

    for model in results:
        if all_done():
            break

        uid        = model["uid"]
        face_count = model.get("faceCount", None)
        name       = model["name"]
        safe_name  = safe_filename(name)

        if face_count is None:
            continue

        # Determine which tier this asset belongs to
        matched_tier = None
        for tier, s in active_tiers().items():
            if s["face_min"] <= face_count < s["face_max"]:
                matched_tier = tier
                break

        if matched_tier is None:
            continue  # not extreme — skip

        s = state[matched_tier]

        if uid in s["completed"]:
            continue

        log(f"[{matched_tier.upper()} {s['downloaded']+1}/{s['target']}] {name} | {face_count:,} faces")

        # Get download link
        dl_resp = safe_get(f"https://api.sketchfab.com/v3/models/{uid}/download", headers)
        dl_data = dl_resp.json()

        if "gltf" not in dl_data:
            log(f"  [SKIP] No GLTF available")
            continue

        gltf_url = dl_data["gltf"]["url"]
        zip_path = os.path.join(s["folder"], f"{safe_name}_{uid}.zip")

        # Download
        try:
            with requests.get(gltf_url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            log(f"  [ERROR] Download failed: {e}")
            continue

        # Extract
        extract_dir = os.path.join(s["folder"], f"{safe_name}_{uid}")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            os.remove(zip_path)
            log(f"  Extracted → {extract_dir}")
        except zipfile.BadZipFile:
            log(f"  [SKIP] Bad zip")
            continue

        # Mark complete
        with open(s["completed_path"], "a") as f:
            f.write(uid + "\n")
        s["completed"].add(uid)
        s["downloaded"] += 1

        log(f"  EXTREME: {state['extreme']['downloaded']}/{state['extreme']['target']}")
        time.sleep(random.uniform(1.5, 4.0))

    # Save shared cursor after every page
    with open(shared_cursor_path, "w") as f:
        f.write(shared_cursor or "")

log("=" * 60)
for tier, s in state.items():
    log(f"  {tier.upper():8s} — downloaded {s['downloaded']}/{s['target']}")
log("=" * 60)
log("Done.")