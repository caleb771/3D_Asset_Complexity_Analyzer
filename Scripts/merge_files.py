import pandas as pd
import os
from pathlib import Path

# Define the folder path
folder = r"C:\Users\USER PC\Downloads\demo"

# Candidate filenames to look for in the demo folder
FEAT_CANDS = ["features_v2.csv", "featuresv2.csv", "features_new.csv", "features.csv"]
REND_CANDS = ["render_times_new.csv", "render_times.csv"]


def find_file(candidates):
    for name in candidates:
        p = Path(folder) / name
        if p.exists():
            return str(p)
    return None


feat_path = find_file(FEAT_CANDS)
rend_path = find_file(REND_CANDS)

if not feat_path or not rend_path:
    raise SystemExit(f"Missing feature or render CSV in {folder}.\n" +
                     f"Looked for: {FEAT_CANDS} and {REND_CANDS}")

# Load CSVs
F = pd.read_csv(feat_path)
R = pd.read_csv(rend_path)

# Required columns check
for col in ("asset_name", "file_name"):
    if col not in F.columns:
        raise SystemExit(f"Feature file missing column: {col}")
    if col not in R.columns:
        raise SystemExit(f"Render file missing column: {col}")

# Normalize join keys: use filename (name only), lowercase, strip whitespace
F["file_nm_norm"] = F["file_name"].astype(str).apply(lambda x: Path(x).name.lower().strip())
R["file_nm_norm"] = R["file_name"].astype(str).apply(lambda x: Path(x).name.lower().strip())
F["asset_nm_norm"] = F["asset_name"].astype(str).str.lower().str.strip()
R["asset_nm_norm"] = R["asset_name"].astype(str).str.lower().str.strip()

# Aggregate duplicate render rows (e.g., multiple runs): take mean render time and mesh_objects
agg_funcs = {
    "render_time_seconds": "mean",
    "mesh_objects": "mean",
    "rendered_at": "first",
    "file_name": "first",
    "asset_name": "first",
}
R_agg = (
    R.groupby(["asset_nm_norm", "file_nm_norm"], as_index=False)
    .agg(agg_funcs)
)

# Deduplicate features (keep first occurrence when multiple feature rows exist)
F_dedup = F.drop_duplicates(subset=["asset_nm_norm", "file_nm_norm"], keep="first")

# Merge (inner join to keep only matched rows)
merged = pd.merge(
    F_dedup,
    R_agg,
    on=["asset_nm_norm", "file_nm_norm"],
    how="inner",
)

# Save merged dataset
output_excel = Path(folder) / "merged_features_render_times_v2.xlsx"
output_csv = Path(folder) / "merged_features_render_times_v2"
".csv"

merged.to_excel(output_excel, index=False)
merged.to_csv(output_csv, index=False)

print("✅ Merged dataset created!")
print("Excel file saved at:", output_excel)
print("CSV file saved at:", output_csv)
print("Rows:", merged.shape[0], "Columns:", merged.shape[1])

# Print basic diagnostics
print('\nDiagnostics:')
print('Original feature rows:', len(F))
print('Original render rows :', len(R))
print('Feature deduped rows :', len(F_dedup))
print('Render aggregated rows:', len(R_agg))

left_only = pd.merge(F_dedup, R_agg, on=["asset_nm_norm", "file_nm_norm"], how="left", indicator=True)
left_only_count = left_only[left_only["_merge"] == "left_only"].shape[0]
right_only = pd.merge(F_dedup, R_agg, on=["asset_nm_norm", "file_nm_norm"], how="right", indicator=True)
right_only_count = right_only[right_only["_merge"] == "right_only"].shape[0]
print('Features without renders (approx):', left_only_count)
print('Renders without features (approx):', right_only_count)
