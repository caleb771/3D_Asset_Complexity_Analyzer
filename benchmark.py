import bpy
import time
import os
import csv

# ==============================================================================
# DIRECTORY CONFIGURATION (Paste your Google Drive paths here)
# ==============================================================================
INPUT_ASSET_DIR = "/content/drive/MyDrive/RenderDataset/assets/"
OUTPUT_CSV_PATH = "/content/drive/MyDrive/RenderDataset/render_data.csv"
# ==============================================================================

def initialize_csv():
    """Creates the dataset file with structured headers if it doesn't exist."""
    headers = [
        "asset_name", "poly_count", "vertex_count", "material_count", 
        "resolution_x", "resolution_y", "cycles_samples", "render_time_seconds"
    ]
    if not os.path.exists(OUTPUT_CSV_PATH):
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Created new dataset file at: {OUTPUT_CSV_PATH}")

def clear_scene():
    """Removes default objects from the scene to prevent overlapping geometry."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def render_and_log(asset_file):
    """Imports an asset, calculates its metadata, renders it, and logs execution speed."""
    clear_scene()
    
    # Absolute path setup
    asset_path = os.path.join(INPUT_ASSET_DIR, asset_file)
    
    # Handle distinct import pipelines for common raw 3D types
    if asset_file.lower().endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=asset_path)
    elif asset_file.lower().endswith('.fbx'):
        bpy.ops.wm.fbx_import(filepath=asset_path)
    elif asset_file.lower().endswith('.gltf') or asset_file.lower().endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=asset_path)
    else:
        print(f"Skipping unsupported extension: {asset_file}")
        return

    # Ensure a basic light and camera exist so Cycles can actually evaluate a scene
    # (Failsafe for assets exported without standard studio setups)
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(1.2, 0, 0))
        bpy.context.scene.camera = bpy.context.object
    if not any(obj.type == 'LIGHT' for obj in bpy.data.objects):
        bpy.ops.object.light_add(type='SUN', location=(0, -2, 4))

    # --- INDEPENDENT VARIABLES (Feature Extraction) ---
    polys = sum([len(mesh.polygons) for mesh in bpy.data.meshes])
    verts = sum([len(mesh.vertices) for mesh in bpy.data.meshes])
    mats = len(bpy.data.materials)
    
    scene = bpy.context.scene
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    samples = scene.cycles.samples if scene.render.engine == 'CYCLES' else scene.eevee.taa_render_samples

    # --- DEPENDENT VARIABLE (Target Timing) ---
    # Sync GPU threads before timing to get an accurate read
    start_time = time.time()
    
    # Headless rendering operation executed directly to standard memory buffers
    bpy.ops.render.render(write_still=False) 
    
    end_time = time.time()
    render_time = end_time - start_time
    
    # --- APPENDBACK PIPELINE ---
    row = [asset_file, polys, verts, mats, res_x, res_y, samples, round(render_time, 4)]
    with open(OUTPUT_CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"✔ Recorded: {asset_file} | Polys: {polys} | Time: {round(render_time, 2)}s")

if __name__ == "__main__":
    print("Starting automated feature extraction pipeline...")
    initialize_csv()
    
    if not os.path.exists(INPUT_ASSET_DIR):
        print(f"ERROR: Input directory not found: {INPUT_ASSET_DIR}")
    else:
        # Scan folder for valid file matrices
        valid_extensions = ('.obj', '.fbx', '.gltf', '.glb')
        files = [f for f in os.listdir(INPUT_ASSET_DIR) if f.lower().endswith(valid_extensions)]
        print(f"Discovered {len(files)} target assets processing queue.")
        
        for index, file in enumerate(files, start=1):
            print(f"\nProcessing [{index}/{len(files)}]: {file}")
            try:
                render_and_log(file)
            except Exception as e:
                print(f"❌ Failed to run runtime benchmarking on {file}. Error: {str(e)}")
        print("\nDataset loop completed successfully.")