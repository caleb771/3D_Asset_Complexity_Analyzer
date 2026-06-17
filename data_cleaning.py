import pandas as pd
import numpy as np

csv_path = r"C:\Users\USER PC\Downloads\demo\merged_features_render_times_v2.csv"

# Try reading with UTF-8 first, fall back to Windows-1252 / latin-1 if necessary
try:
	df = pd.read_csv(csv_path)
	used_encoding = 'utf-8 (default)'
except UnicodeDecodeError:
	try:
		df = pd.read_csv(csv_path, encoding='cp1252')
		used_encoding = 'cp1252'
	except UnicodeDecodeError:
		df = pd.read_csv(csv_path, encoding='latin-1')
		used_encoding = 'latin-1'

print(f"Read CSV using encoding: {used_encoding}")
df = df.dropna(subset=["render_time_seconds"])

print(f"Before dedup: {len(df)} rows")

# Remove duplicates — same asset name and same face count
# Some merged CSVs have suffixes like asset_name_x / asset_name_y
# normalize to a single `asset_name` and `file_name` if necessary
if "asset_name" not in df.columns:
	if "asset_name_x" in df.columns:
		df["asset_name"] = df["asset_name_x"]
	elif "asset_name_y" in df.columns:
		df["asset_name"] = df["asset_name_y"]

if "file_name" not in df.columns:
	if "file_name_x" in df.columns:
		df["file_name"] = df["file_name_x"]
	elif "file_name_y" in df.columns:
		df["file_name"] = df["file_name_y"]

if "face_count" not in df.columns:
	raise SystemExit("Expected column 'face_count' not found in CSV")

df = df.drop_duplicates(subset=["asset_name", "face_count"], keep="first")

print(f"After dedup:  {len(df)} rows")

# Check complexity distribution after dedup
print(f"\nComplexity distribution:")
print(df["complexity_tier"].value_counts())

# Save cleaned dataset
df.to_csv(r"C:\Users\USER PC\Downloads\demo\final_model\dataset_clean_new_v3.csv", index=False)
print("\nSaved to dataset_clean_new_v2.csv")