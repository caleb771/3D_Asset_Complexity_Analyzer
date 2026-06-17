"""
get_render_times.py
====================
Automatically finds all 3D assets in your demo folder,
opens Blender once per asset, renders it, records the
render time, and saves everything to render_times.csv.

Run from VS Code terminal:
    python get_render_times.py

Or with explicit Python path:
    & "C:/Users/USER PC/AppData/Local/Programs/Python/Python311/python.exe" get_render_times.py
"""

import subprocess
import sys
import time
from pathlib import Path

# ── CONFIG ── edit these if your paths are different ─────────────────────────
BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
TIMER_SCRIPT = r"C:\Users\USER PC\Downloads\demo\files (4)\blender_render_timer.py"
DEMO_FOLDER  = r"C:\Users\USER PC\Downloads\demo\demo_new"
OUTPUT_CSV   = r"C:\Users\USER PC\Downloads\demo\render_times_new.csv"
# ─────────────────────────────────────────────────────────────────────────────

demo = Path(DEMO_FOLDER)
output = Path(OUTPUT_CSV)

# Find all scene files — scene.gltf, scene.glb, .fbx in source subfolders
scene_files = []

# scene.gltf — one per Sketchfab asset folder
scene_files += list(demo.rglob("scene.gltf"))

# standalone GLB files (e.g. mutant-animal-talk/source/*.glb)
scene_files += list(demo.rglob("*.glb"))

# FBX files in source subfolders
scene_files += list(demo.rglob("*.fbx"))

# OBJ files
scene_files += list(demo.rglob("*.obj"))

# Remove duplicates and sort
scene_files = sorted(set(scene_files))

if not scene_files:
    print(f"[ERROR] No 3D files found in {DEMO_FOLDER}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"  3D Asset Render Timer")
print(f"{'='*60}")
print(f"  Assets found : {len(scene_files)}")
print(f"  Output CSV   : {OUTPUT_CSV}")
print(f"  Blender      : {BLENDER}")
print(f"{'='*60}\n")

# Check Blender exists
if not Path(BLENDER).exists():
    print(f"[ERROR] Blender not found at: {BLENDER}")
    print("        Update the BLENDER path at the top of this script.")
    sys.exit(1)

# Check timer script exists
if not Path(TIMER_SCRIPT).exists():
    print(f"[ERROR] blender_render_timer.py not found at: {TIMER_SCRIPT}")
    print("        Update the TIMER_SCRIPT path at the top of this script.")
    sys.exit(1)

success = []
failed  = []
total   = len(scene_files)

for i, scene in enumerate(scene_files):
    asset_name = (
        scene.parent.name
        if scene.name in {"scene.gltf", "scene.glb", "scene.fbx"}
        else scene.stem
    )

    print(f"[{i+1:02d}/{total}] {asset_name} ({scene.name})")

    t_wall = time.time()

    result = subprocess.run(
        [
            BLENDER,
            "--background",
            "--python", TIMER_SCRIPT,
            "--",
            "--input",  str(scene),
            "--output", OUTPUT_CSV,
        ],
        capture_output=False,   # show Blender output in terminal
        timeout=3600,           # 1 hour max per asset
    )

    elapsed = round(time.time() - t_wall, 1)

    if result.returncode == 0:
        success.append(asset_name)
        print(f"       ✓ Done in {elapsed}s wall time\n")
    else:
        failed.append(asset_name)
        print(f"       ✗ FAILED (return code {result.returncode})\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  COMPLETE")
print(f"  Rendered : {len(success)} assets")
print(f"  Failed   : {len(failed)} assets")
print(f"  Output   : {OUTPUT_CSV}")
print(f"{'='*60}\n")

if failed:
    print("Failed assets:")
    for name in failed:
        print(f"  - {name}")
    # Write failed list to file
    fail_log = Path(DEMO_FOLDER) / "render_failed.txt"
    fail_log.write_text("\n".join(failed))
    print(f"\nFailed list saved to: {fail_log}")
