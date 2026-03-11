"""
Blender Asset Feature Extractor (bpy)
======================================
Run this script INSIDE Blender's Python environment to extract features
that are only accessible via the bpy API: subdivision modifiers, shader
node graphs, UV maps, particle systems, and render settings.

Usage (headless):
    blender --background scene.blend --python blender_extractor.py -- --output features.csv

Usage (from Blender's Script Editor):
    Simply open and run this file. Results print to the console and
    write to features.csv in the same directory as the .blend file.

The output CSV is compatible with feature_extractor.py — columns are
named identically so both can be concatenated into one dataset.
"""

import sys
import os
import csv
import time
import json
from pathlib import Path
from datetime import datetime

# ── detect if running inside Blender ────────────────────────────────────────
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] This script must be run inside Blender's Python environment.")
    print("        Command: blender --background file.blend --python blender_extractor.py")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# GEOMETRY FEATURES (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_geometry_features(scene) -> dict:
    mesh_objects = [o for o in scene.objects if o.type == "MESH"]

    total_verts  = 0
    total_faces  = 0
    total_edges  = 0

    for obj in mesh_objects:
        # Apply modifiers to get real evaluated mesh
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj  = obj.evaluated_get(depsgraph)
            mesh      = eval_obj.to_mesh()
            total_verts += len(mesh.vertices)
            total_faces += len(mesh.polygons)
            total_edges += len(mesh.edges)
            eval_obj.to_mesh_clear()
        except Exception:
            # Fallback to base mesh data
            if obj.data:
                total_verts += len(obj.data.vertices)
                total_faces += len(obj.data.polygons)
                total_edges += len(obj.data.edges)

    return {
        "mesh_count":   len(mesh_objects),
        "vertex_count": total_verts,
        "face_count":   total_faces,
        "edge_count":   total_edges,
    }


# ════════════════════════════════════════════════════════════════════════════
# MODIFIER FEATURES (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_modifier_features(scene) -> dict:
    mesh_objects = [o for o in scene.objects if o.type == "MESH"]

    has_subdivision      = False
    max_subdiv_level     = 0
    has_displacement     = False
    has_particles        = False
    has_boolean          = False
    total_modifier_count = 0

    for obj in mesh_objects:
        for mod in obj.modifiers:
            total_modifier_count += 1
            if mod.type == "SUBSURF":
                has_subdivision = True
                max_subdiv_level = max(max_subdiv_level, getattr(mod, "levels", 0))
            if mod.type == "DISPLACE":
                has_displacement = True
            if mod.type == "PARTICLE_SYSTEM":
                has_particles = True
            if mod.type == "BOOLEAN":
                has_boolean = True

    return {
        "has_subdivision":      has_subdivision,
        "max_subdiv_level":     max_subdiv_level,
        "has_displacement":     has_displacement,
        "has_particles":        has_particles,
        "has_boolean":          has_boolean,
        "total_modifier_count": total_modifier_count,
    }


# ════════════════════════════════════════════════════════════════════════════
# MATERIAL / SHADER FEATURES (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_material_features() -> dict:
    materials = list(bpy.data.materials)
    mat_count = len(materials)

    shader_node_count    = 0
    has_principled_bsdf  = False
    has_subsurface       = False
    has_volume           = False
    has_emission         = False

    for mat in materials:
        if mat.use_nodes and mat.node_tree:
            nodes = mat.node_tree.nodes
            shader_node_count += len(nodes)
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    has_principled_bsdf = True
                    # Check subsurface weight
                    sss_input = node.inputs.get("Subsurface Weight") or node.inputs.get("Subsurface")
                    if sss_input and sss_input.default_value > 0:
                        has_subsurface = True
                if node.type in {"VOLUME_ABSORPTION", "VOLUME_SCATTER", "PRINCIPLED_VOLUME"}:
                    has_volume = True
                if node.type == "EMISSION":
                    has_emission = True

    return {
        "material_count":       mat_count,
        "shader_node_count":    shader_node_count,
        "has_principled_bsdf":  has_principled_bsdf,
        "has_subsurface":       has_subsurface,
        "has_volume_shader":    has_volume,
        "has_emission_shader":  has_emission,
    }


# ════════════════════════════════════════════════════════════════════════════
# TEXTURE FEATURES (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_texture_features() -> dict:
    images = [img for img in bpy.data.images if img.size[0] > 0 and img.size[1] > 0]

    texture_count        = len(images)
    total_pixels         = sum(img.size[0] * img.size[1] for img in images)
    total_memory_mb      = sum(
        img.size[0] * img.size[1] * img.channels / (1024 ** 2)
        for img in images
    )
    max_w = max((img.size[0] for img in images), default=0)
    max_h = max((img.size[1] for img in images), default=0)

    return {
        "texture_count":           texture_count,
        "total_texture_pixels":    total_pixels,
        "total_texture_memory_mb": round(total_memory_mb, 4),
        "max_texture_width":       max_w,
        "max_texture_height":      max_h,
    }


# ════════════════════════════════════════════════════════════════════════════
# LIGHT FEATURES (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_light_features(scene) -> dict:
    lights = [o for o in scene.objects if o.type == "LIGHT"]

    light_types = {}
    has_hdri     = False

    for obj in lights:
        lt = obj.data.type
        light_types[lt] = light_types.get(lt, 0) + 1

    # Check for HDRI (world environment texture)
    world = scene.world
    if world and world.use_nodes and world.node_tree:
        for node in world.node_tree.nodes:
            if node.type == "TEX_ENVIRONMENT":
                has_hdri = True

    return {
        "light_count":        len(lights),
        "light_point_count":  light_types.get("POINT", 0),
        "light_sun_count":    light_types.get("SUN", 0),
        "light_area_count":   light_types.get("AREA", 0),
        "light_spot_count":   light_types.get("SPOT", 0),
        "has_hdri":           has_hdri,
    }


# ════════════════════════════════════════════════════════════════════════════
# UV FEATURES (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_uv_features(scene) -> dict:
    mesh_objects = [o for o in scene.objects if o.type == "MESH" and o.data]
    uv_counts = [len(o.data.uv_layers) for o in mesh_objects]

    return {
        "total_uv_maps":   sum(uv_counts),
        "max_uv_maps":     max(uv_counts, default=0),
        "objects_with_uv": sum(1 for c in uv_counts if c > 0),
    }


# ════════════════════════════════════════════════════════════════════════════
# RENDER SETTINGS (bpy)
# ════════════════════════════════════════════════════════════════════════════

def get_render_settings(scene) -> dict:
    render  = scene.render
    cycles  = scene.cycles if hasattr(scene, "cycles") else None

    settings = {
        "render_engine":        render.engine,
        "resolution_x":         render.resolution_x,
        "resolution_y":         render.resolution_y,
        "resolution_pct":       render.resolution_percentage,
        "output_pixels":        (render.resolution_x * render.resolution_y *
                                  render.resolution_percentage // 100),
    }

    if cycles and render.engine == "CYCLES":
        settings["cycles_samples"]    = getattr(cycles, "samples", None)
        settings["cycles_max_bounces"] = getattr(cycles, "max_bounces", None)
        settings["cycles_use_denoising"] = getattr(cycles, "use_denoising", False)
        settings["cycles_device"]     = getattr(cycles, "device", None)
    else:
        settings["cycles_samples"]    = None
        settings["cycles_max_bounces"] = None
        settings["cycles_use_denoising"] = None
        settings["cycles_device"]     = None

    return settings


# ════════════════════════════════════════════════════════════════════════════
# COMPLEXITY TIER LABELLING
# ════════════════════════════════════════════════════════════════════════════

def assign_complexity_tier(face_count: int) -> str:
    if face_count < 10_000:
        return "low"
    elif face_count < 100_000:
        return "mid"
    elif face_count < 1_000_000:
        return "high"
    return "extreme"


# ════════════════════════════════════════════════════════════════════════════
# TIMED RENDER (optional — records actual render time)
# ════════════════════════════════════════════════════════════════════════════

def timed_render(output_path: str = "/tmp/render_output.png") -> float:
    """
    Perform a single render and return wall-clock time in seconds.
    Set output_path to /dev/null or a temp path to avoid keeping files.
    """
    scene = bpy.context.scene
    original_output = scene.render.filepath

    scene.render.filepath = output_path
    t_start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    render_time = time.perf_counter() - t_start

    scene.render.filepath = original_output
    return round(render_time, 4)


# ════════════════════════════════════════════════════════════════════════════
# FULL EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_all(do_render: bool = False) -> dict:
    """
    Run all feature extractors and return a single flat dict.
    Set do_render=True to also time an actual render (slow).
    """
    scene     = bpy.context.scene
    blend_path = Path(bpy.data.filepath)

    row = {}

    # File metadata
    row["file_name"]      = blend_path.name
    row["file_format"]    = "blend"
    row["file_size_mb"]   = round(blend_path.stat().st_size / (1024**2), 4) if blend_path.exists() else None
    row["file_size_bytes"] = blend_path.stat().st_size if blend_path.exists() else None

    # All feature groups
    row.update(get_geometry_features(scene))
    row.update(get_modifier_features(scene))
    row.update(get_material_features())
    row.update(get_texture_features())
    row.update(get_light_features(scene))
    row.update(get_uv_features(scene))
    row.update(get_render_settings(scene))

    # Labels
    row["complexity_tier"] = assign_complexity_tier(row.get("face_count", 0))
    row["extracted_at"]    = datetime.utcnow().isoformat()

    # Optional: time a real render
    if do_render:
        print("[INFO] Running timed render (this may take a while)...")
        row["render_time_seconds"] = timed_render()
        print(f"[INFO] Render time: {row['render_time_seconds']}s")
    else:
        row["render_time_seconds"] = None

    return row


# ════════════════════════════════════════════════════════════════════════════
# WRITE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def write_csv(row: dict, output_path: Path, append: bool = True) -> None:
    file_exists = output_path.exists()
    mode = "a" if (append and file_exists) else "w"
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not (append and file_exists):
            writer.writeheader()
        writer.writerow(row)
    print(f"[INFO] Written to {output_path}")


def write_json(row: dict, output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    print(f"[INFO] JSON written to {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Parse args passed after "--" in the blender command
    argv = sys.argv
    script_args = argv[argv.index("--") + 1:] if "--" in argv else []

    output_csv  = Path(script_args[0]) if len(script_args) > 0 else Path("features.csv")
    do_render   = "--render" in script_args
    append      = "--no-append" not in script_args

    print("[INFO] 3D Asset Complexity Analyzer — Blender Extractor")
    print(f"[INFO] Scene: {bpy.data.filepath or 'unsaved'}")
    print(f"[INFO] Output: {output_csv}")
    print(f"[INFO] Timed render: {do_render}")

    features = extract_all(do_render=do_render)

    # Print summary
    print("\n── Extracted Features ──────────────────────────────────")
    for k, v in features.items():
        print(f"  {k:<35} {v}")
    print("────────────────────────────────────────────────────────\n")

    write_csv(features, output_csv, append=append)
    write_json(features, output_csv.with_suffix(".json"))
