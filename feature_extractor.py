"""
3D Asset Complexity Analyzer — Feature Extraction Pipeline
===========================================================
Extracts mesh statistics and material/texture metadata from 3D asset files
and writes one row per asset to a structured CSV dataset.

Supported formats (via trimesh):
    OBJ, FBX, PLY, STL, GLTF/GLB, OFF, 3DS, DAE (Collada)

Blender-specific extraction (requires running inside Blender's Python):
    See blender_extractor.py for bpy-based extraction of subdivision,
    shader graphs, and UV maps.

Dependencies:
    pip install trimesh numpy pandas Pillow open3d tqdm

Usage:
    # Extract features from all assets in a folder
    python feature_extractor.py --input ./assets --output dataset.csv

    # Single file
    python feature_extractor.py --input ./assets/my_model.obj --output dataset.csv

    # Recursive search through subdirectories
    python feature_extractor.py --input ./assets --output dataset.csv --recursive

    # Append to an existing CSV instead of overwriting
    python feature_extractor.py --input ./assets --output dataset.csv --append
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── optional dependencies ────────────────────────────────────────────────────
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[WARN] trimesh not installed. Run: pip install trimesh")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed. Texture stats will be skipped. Run: pip install Pillow")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extraction.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".obj", ".fbx", ".ply", ".stl", ".glb", ".gltf",
    ".off", ".3ds", ".dae", ".blend"
}

COMPLEXITY_THRESHOLDS = {
    "low":    (0,       10_000),
    "mid":    (10_000,  100_000),
    "high":   (100_000, 1_000_000),
    "extreme":(1_000_000, float("inf")),
}


# ════════════════════════════════════════════════════════════════════════════
# GEOMETRY FEATURES
# ════════════════════════════════════════════════════════════════════════════

def extract_geometry_features(mesh) -> dict:
    """
    Extract polygon, vertex, edge and topology statistics from a trimesh object.
    Works on both trimesh.Trimesh and trimesh.Scene (multi-mesh) objects.
    """
    features = {}

    # Flatten scenes into a single merged mesh for aggregate stats
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            return _empty_geometry_features()
        try:
            combined = trimesh.util.concatenate(geometries)
        except Exception:
            combined = geometries[0]          # fallback to first mesh
        mesh_count = len(geometries)
    else:
        combined = mesh
        mesh_count = 1

    features["mesh_count"]    = mesh_count
    features["vertex_count"]  = len(combined.vertices)
    features["face_count"]    = len(combined.faces)
    features["edge_count"]    = len(combined.edges_unique) if hasattr(combined, "edges_unique") else 0

    # Derived density metrics
    bb_vol = float(combined.bounding_box.volume) if combined.bounding_box else 1.0
    features["bounding_box_volume"]   = round(bb_vol, 6)
    features["vertex_density"]        = round(features["vertex_count"] / max(bb_vol, 1e-9), 4)
    features["polygon_density"]       = round(features["face_count"]   / max(bb_vol, 1e-9), 4)

    # Bounding box dimensions
    extents = combined.bounding_box.extents if combined.bounding_box else [0, 0, 0]
    features["bbox_x"] = round(float(extents[0]), 6)
    features["bbox_y"] = round(float(extents[1]), 6)
    features["bbox_z"] = round(float(extents[2]), 6)

    # Surface area
    try:
        features["surface_area"] = round(float(combined.area), 6)
    except Exception:
        features["surface_area"] = None

    # Topology / quality flags
    features["is_watertight"]      = bool(combined.is_watertight)
    features["is_winding_consistent"] = bool(combined.is_winding_consistent)
    try:
        features["euler_number"]   = int(combined.euler_number)
    except Exception:
        features["euler_number"]   = None

    # Duplicate / degenerate face detection
    try:
        features["duplicate_faces"]   = int(len(combined.faces) - len(np.unique(combined.faces, axis=0)))
    except Exception:
        features["duplicate_faces"]   = None

    try:
        degen = trimesh.triangles.area(combined.triangles) < 1e-10
        features["degenerate_faces"]  = int(np.sum(degen))
    except Exception:
        features["degenerate_faces"]  = None

    return features


def _empty_geometry_features() -> dict:
    keys = [
        "mesh_count", "vertex_count", "face_count", "edge_count",
        "bounding_box_volume", "vertex_density", "polygon_density",
        "bbox_x", "bbox_y", "bbox_z", "surface_area",
        "is_watertight", "is_winding_consistent", "euler_number",
        "duplicate_faces", "degenerate_faces"
    ]
    return {k: None for k in keys}


# ════════════════════════════════════════════════════════════════════════════
# MATERIAL / TEXTURE FEATURES
# ════════════════════════════════════════════════════════════════════════════

def extract_material_features(mesh, asset_path: Path) -> dict:
    """
    Count materials and, where texture file paths can be resolved,
    measure texture dimensions and compute memory estimates.
    """
    features = {
        "material_count":           0,
        "texture_count":            0,
        "total_texture_pixels":     0,
        "total_texture_memory_mb":  0.0,
        "max_texture_width":        0,
        "max_texture_height":       0,
        "has_normal_map":           False,
        "has_displacement_map":     False,
    }

    # ── material count ───────────────────────────────────────────────────
    try:
        if isinstance(mesh, trimesh.Scene):
            mat_names = set()
            for geom in mesh.geometry.values():
                if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                    mat_names.add(id(geom.visual.material))
            features["material_count"] = len(mat_names) if mat_names else 0
        elif hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
            features["material_count"] = 1
    except Exception:
        pass

    # ── texture resolution from neighbouring texture files ───────────────
    if not HAS_PIL:
        return features

    asset_dir = asset_path.parent
    texture_extensions = {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".exr", ".bmp"}
    texture_files = [
        f for f in asset_dir.iterdir()
        if f.suffix.lower() in texture_extensions
    ]

    normal_keywords      = {"normal", "nrm", "nor", "_n.", "_n_"}
    displacement_keywords = {"disp", "displacement", "height", "bump"}

    pixel_total  = 0
    memory_total = 0.0
    max_w, max_h = 0, 0

    for tex_path in texture_files:
        name_lower = tex_path.stem.lower()
        if any(kw in name_lower for kw in normal_keywords):
            features["has_normal_map"] = True
        if any(kw in name_lower for kw in displacement_keywords):
            features["has_displacement_map"] = True

        try:
            with Image.open(tex_path) as img:
                w, h = img.size
                channels = len(img.getbands())
                pixels   = w * h
                pixel_total  += pixels
                memory_total += (pixels * channels) / (1024 ** 2)   # MB uncompressed
                max_w = max(max_w, w)
                max_h = max(max_h, h)
                features["texture_count"] += 1
        except Exception:
            pass

    features["total_texture_pixels"]    = pixel_total
    features["total_texture_memory_mb"] = round(memory_total, 4)
    features["max_texture_width"]       = max_w
    features["max_texture_height"]      = max_h

    return features


# ════════════════════════════════════════════════════════════════════════════
# FILE / META FEATURES
# ════════════════════════════════════════════════════════════════════════════

def extract_file_features(asset_path: Path) -> dict:
    stat = asset_path.stat()
    return {
        "file_name":       asset_path.name,
        "file_format":     asset_path.suffix.lower().lstrip("."),
        "file_size_mb":    round(stat.st_size / (1024 ** 2), 4),
        "file_size_bytes": stat.st_size,
    }


# ════════════════════════════════════════════════════════════════════════════
# COMPLEXITY TIER LABELLING
# ════════════════════════════════════════════════════════════════════════════

def assign_complexity_tier(face_count: int) -> str:
    for tier, (lo, hi) in COMPLEXITY_THRESHOLDS.items():
        if lo <= face_count < hi:
            return tier
    return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def extract_features(asset_path: Path) -> dict | None:
    """
    Full feature extraction pipeline for a single asset file.
    Returns a flat dict of all features, or None on unrecoverable error.
    """
    if not HAS_TRIMESH:
        log.error("trimesh is required. Install with: pip install trimesh")
        return None

    log.info(f"Processing: {asset_path.name}")
    t_start = time.perf_counter()

    row = {}

    # 1. File metadata
    row.update(extract_file_features(asset_path))

    # 2. Load mesh
    try:
        mesh = trimesh.load(str(asset_path), force="mesh", process=False)
    except Exception as e:
        log.warning(f"  [SKIP] Failed to load {asset_path.name}: {e}")
        return None

    if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
        log.warning(f"  [SKIP] Empty mesh: {asset_path.name}")
        return None

    # 3. Geometry features
    try:
        row.update(extract_geometry_features(mesh))
    except Exception as e:
        log.warning(f"  [WARN] Geometry extraction partial: {e}")
        row.update(_empty_geometry_features())

    # 4. Material / texture features
    try:
        row.update(extract_material_features(mesh, asset_path))
    except Exception as e:
        log.warning(f"  [WARN] Material extraction failed: {e}")

    # 5. Derived / label fields
    face_count = row.get("face_count") or 0
    row["complexity_tier"]      = assign_complexity_tier(face_count)
    row["extraction_time_s"]    = round(time.perf_counter() - t_start, 4)
    row["extracted_at"]         = datetime.utcnow().isoformat()

    log.info(
        f"  → {face_count:,} faces | "
        f"{row.get('vertex_count', 0):,} verts | "
        f"tier={row['complexity_tier']} | "
        f"{row['extraction_time_s']}s"
    )
    return row


# ════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def collect_asset_paths(input_path: Path, recursive: bool) -> list[Path]:
    """Return all supported 3D asset files under input_path."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    paths = [
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def run_batch(
    input_path: Path,
    output_csv: Path,
    recursive: bool = False,
    append: bool = False,
) -> pd.DataFrame:
    """
    Extract features from all assets found under input_path and
    write results to output_csv.
    """
    asset_paths = collect_asset_paths(input_path, recursive)
    if not asset_paths:
        log.warning(f"No supported 3D files found in: {input_path}")
        return pd.DataFrame()

    log.info(f"Found {len(asset_paths)} asset(s) to process.")

    iterator = tqdm(asset_paths, desc="Extracting") if HAS_TQDM else asset_paths
    rows = []
    failed = []

    for path in iterator:
        try:
            row = extract_features(path)
            if row:
                rows.append(row)
        except Exception:
            log.error(f"  [ERROR] Unhandled exception on {path.name}:\n{traceback.format_exc()}")
            failed.append(str(path))

    if not rows:
        log.warning("No features extracted. Check asset files and dependencies.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── column ordering ──────────────────────────────────────────────────
    priority_cols = [
        "file_name", "file_format", "file_size_mb",
        "mesh_count", "vertex_count", "face_count", "edge_count",
        "surface_area", "bounding_box_volume", "vertex_density", "polygon_density",
        "bbox_x", "bbox_y", "bbox_z",
        "is_watertight", "is_winding_consistent", "euler_number",
        "duplicate_faces", "degenerate_faces",
        "material_count", "texture_count",
        "total_texture_pixels", "total_texture_memory_mb",
        "max_texture_width", "max_texture_height",
        "has_normal_map", "has_displacement_map",
        "complexity_tier",
        "file_size_bytes", "extraction_time_s", "extracted_at",
    ]
    remaining = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + remaining]

    # ── write CSV ────────────────────────────────────────────────────────
    mode   = "a" if append and output_csv.exists() else "w"
    header = not (append and output_csv.exists())
    df.to_csv(output_csv, mode=mode, header=header, index=False)

    # Summary
    log.info("─" * 60)
    log.info(f"Extracted : {len(rows)} assets")
    log.info(f"Failed    : {len(failed)} assets")
    log.info(f"Output    : {output_csv.resolve()}")
    log.info(f"Columns   : {len(df.columns)}")
    log.info("─" * 60)

    # Complexity tier distribution
    if "complexity_tier" in df.columns:
        log.info("Complexity distribution:")
        for tier, count in df["complexity_tier"].value_counts().items():
            log.info(f"  {tier:10s}: {count} assets")

    if failed:
        fail_log = output_csv.with_name("failed_assets.txt")
        fail_log.write_text("\n".join(failed))
        log.warning(f"Failed asset paths written to: {fail_log}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# QUICK SUMMARY REPORT (printed to stdout)
# ════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    numeric_cols = [
        "face_count", "vertex_count", "file_size_mb",
        "total_texture_pixels", "total_texture_memory_mb"
    ]
    available = [c for c in numeric_cols if c in df.columns]

    print("\n" + "═" * 60)
    print("  DATASET SUMMARY")
    print("═" * 60)
    print(df[available].describe().round(2).to_string())
    print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# SINGLE-FILE CONVENIENCE FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def analyze_single(file_path: str) -> dict | None:
    """
    Convenience wrapper for use in notebooks or other scripts.

    Example:
        from feature_extractor import analyze_single
        features = analyze_single("my_model.obj")
        print(features)
    """
    return extract_features(Path(file_path))


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="3D Asset Complexity Analyzer — Feature Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to a single asset file OR a directory of assets"
    )
    p.add_argument(
        "--output", "-o", default="dataset.csv",
        help="Output CSV file path (default: dataset.csv)"
    )
    p.add_argument(
        "--recursive", "-r", action="store_true",
        help="Search subdirectories recursively for assets"
    )
    p.add_argument(
        "--append", "-a", action="store_true",
        help="Append to existing CSV instead of overwriting"
    )
    p.add_argument(
        "--summary", "-s", action="store_true",
        help="Print a statistical summary of the extracted dataset"
    )
    p.add_argument(
        "--json", "-j", action="store_true",
        help="Also write dataset.json alongside the CSV"
    )
    return p.parse_args()


def main():
    input_path  = Path(r"C:\Users\USER PC\Downloads\demo")
    output_path = Path(r"C:\Users\USER PC\Downloads\demo\features.csv")

    if not input_path.exists():
        log.error(f"Input path does not exist: {input_path}")
        return

    df = run_batch(
        input_path  = input_path,
        output_csv  = output_path,
        recursive   = True,
        append      = False,
    )

    print_summary(df)


main()
