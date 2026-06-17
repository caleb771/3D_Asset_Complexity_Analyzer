import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\USER PC\Downloads\demo\final_model\dataset_clean_new_v3.csv")

# Drop rows with no render time
df = df.dropna(subset=["render_time_seconds"])

print(f"Usable rows: {len(df)}")
print(f"\nComplexity distribution:")
print(df["complexity_tier"].value_counts())

print(f"\nRender time stats:")
print(df["render_time_seconds"].describe())

# Correlation heatmap
numeric = df.select_dtypes(include="number")
plt.figure(figsize=(14, 10))
sns.heatmap(numeric.corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(r"C:\Users\USER PC\Downloads\demo\correlation_heatmap_new_v3.png")
plt.show()

# Render time vs face count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="face_count", y="render_time_seconds", hue="complexity_tier")
plt.xscale("log")
plt.yscale("log")
plt.title("Face Count vs Render Time")
plt.tight_layout()
plt.savefig(r"C:\Users\USER PC\Downloads\demo\face_vs_rendertime_new_v3.png")
plt.show()

# Use the cleaned dataset for summary statistics below (already loaded earlier)
print("SLOWEST 10 ASSETS:")
print(df.nlargest(10, "render_time_seconds")[["asset_name", "face_count", "complexity_tier", "render_time_seconds"]].to_string())

print("\nFASTEST 10 ASSETS:")
print(df.nsmallest(10, "render_time_seconds")[["asset_name", "face_count", "complexity_tier", "render_time_seconds"]].to_string())

print("\nEXTREME TIER ASSETS:")
print(df[df["complexity_tier"] == "extreme"][["asset_name", "face_count", "render_time_seconds"]].to_string())
 
