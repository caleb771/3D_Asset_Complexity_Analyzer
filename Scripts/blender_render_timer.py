"""
blender_render_timer.py
========================
Called automatically by run_pipeline.ps1 or get_render_times.py for each asset.

Imports a GLTF/GLB/FBX/OBJ/PLY file into a fresh Blender scene,
enforces standardized render settings, renders one frame, records
the wall-clock time, and appends one row to render_times.csv.

Usage (called by pipeline):
    blender --background --python blender_render_timer.py \
        -- --input "path/to/scene.gltf" --output "render_times.csv"

Usage (manual single asset test):
    blender --background --python blender_render_timer.py \
        -- --input "C:/Users/USER PC/Downloads/demo/911/scene.gltf" \
           --output "C:/Users/USER PC/Downloads/demo/render_times.csv"
"""

import bpy
import sys
import os
import csv
import time
from pathlib import Path
from datetime import datetime

# ── Parse args after "--" ────────────────────────────────────────────────────
argv  = sys.argv
args  = argv[argv.index("--") + 1:] if "--" in argv else []

input_file = None
output_csv = None

i = 0
while i < len(args):
    if args[i] == "--input" and i + 1 < len(args):
        input_file = args[i + 1]; i += 2
    elif args[i] == "--output" and i + 1 < len(args):
        output_csv = args[i + 1]; i += 2
    else:
        i += 1

if not input_file or not output_csv:
    print("[ERROR] Usage: blender --background --python blender_render_timer.py "
          "-- --input FILE --output CSV")
    sys.exit(1)

input_path  = Path(input_file)
output_path = Path(output_csv)

# Asset name — use parent folder for Sketchfab scene.gltf files
asset_name = (
    input_path.parent.name
    if input_path.name in {"scene.gltf", "scene.glb", "scene.fbx"}
    else input_path.stem
)

print(f"[INFO] Asset   : {asset_name}")
print(f"[INFO] File    : {input_path.name}")
print(f"[INFO] Output  : {output_path}")

# ── Clear default Blender scene ──────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# ── Import asset ─────────────────────────────────────────────────────────────
ext = input_path.suffix.lower()

try:
    if ext in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(input_path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(input_path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(input_path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(input_path))
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(input_path))
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=str(input_path))
    else:
        print(f"[ERROR] Unsupported format: {ext}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
print(f"[INFO] Imported: {len(mesh_objects)} mesh objects")

if not mesh_objects:
    print("[ERROR] No mesh objects after import — skipping")
    sys.exit(1)

# ── Select all imported mesh objects ─────────────────────────────────────────
bpy.ops.object.select_all(action="DESELECT")
for obj in mesh_objects:
    obj.select_set(True)

# ── Add camera ───────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(5, -5, 3))
cam_obj = bpy.context.active_object
cam_obj.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = cam_obj

# Frame all mesh objects in camera view
bpy.ops.object.select_all(action="DESELECT")
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]

# Use camera_to_view_selected if a 3D viewport is available
# Otherwise manually position camera using bounding box
try:
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.view3d.camera_to_view_selected()
            break
except Exception:
    # Fallback: position camera at centroid of all mesh bounding boxes
    import mathutils
    all_verts = []
    for obj in mesh_objects:
        for v in obj.bound_box:
            all_verts.append(obj.matrix_world @ mathutils.Vector(v))
    if all_verts:
        xs = [v.x for v in all_verts]
        ys = [v.y for v in all_verts]
        zs = [v.z for v in all_verts]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        extent = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
        cam_obj.location = (cx + extent, cy - extent, cz + extent * 0.5)
        cam_obj.rotation_euler = (1.1, 0, 0.785)

# ── Add sun light ─────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 3.0

# ── Enforce fixed render settings ────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine                       = "CYCLES"
scene.render.resolution_x                 = 1920
scene.render.resolution_y                 = 1080
scene.render.resolution_percentage        = 100
scene.render.image_settings.file_format   = "PNG"
scene.cycles.samples                      = 128
scene.cycles.max_bounces                  = 12
scene.cycles.use_denoising                = True
scene.cycles.device                       = "CPU"

# Output to a temp file we delete afterwards
temp_output = str(output_path.parent / f"_tmp_render_{asset_name}")
scene.render.filepath = temp_output

print(f"[INFO] Render settings: Cycles CPU | 128 samples | 1920x1080")
print(f"[INFO] Rendering...")

# ── Timed render ─────────────────────────────────────────────────────────────
t_start = time.perf_counter()
bpy.ops.render.render(write_still=True)
render_time = round(time.perf_counter() - t_start, 4)

print(f"[INFO] Render complete: {render_time}s")

# ── Delete temp render output ────────────────────────────────────────────────
for suffix in [".png", ".jpg", ".exr"]:
    tmp = Path(temp_output + suffix)
    if tmp.exists():
        tmp.unlink()

# ── Write to CSV ──────────────────────────────────────────────────────────────
row = {
    "file_name":           input_path.name,
    "asset_name":          asset_name,
    "render_time_seconds": render_time,
    "mesh_objects":        len(mesh_objects),
    "rendered_at":         datetime.utcnow().isoformat(),
}

file_exists = output_path.exists()
with open(output_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"[INFO] Written to {output_path}")
print(f"[INFO] {asset_name}: {render_time}s")
