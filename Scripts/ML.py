import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Load cleaned dataset ──────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\USER PC\Downloads\demo\final_model\dataset_clean_new_v3.csv")
df = df.dropna(subset=["render_time_seconds"])
print(f"Rows: {len(df)}")

# ── Feature engineering ───────────────────────────────────────────────────────
# Log-transform skewed features before feeding to models
log_features = [
    "face_count", "vertex_count", "edge_count",
    "total_texture_pixels", "total_texture_memory_mb",
    "file_size_mb", "surface_area", "bounding_box_volume",
]

for col in log_features:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))

# Feature list — use log-transformed versions of skewed features
feature_cols = [
    "log_face_count", "log_vertex_count", "log_edge_count",
    "log_total_texture_pixels", "log_total_texture_memory_mb",
    "log_file_size_mb", "log_surface_area", "log_bounding_box_volume",
    "mesh_count", "material_count", "texture_count",
    "max_texture_width", "max_texture_height",
    "vertex_density", "polygon_density",
    "has_normal_map", "has_displacement_map", "is_watertight",
]

# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"Features used: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df["render_time_seconds"].copy()

# Fill any remaining nulls
X = X.fillna(0)

# Encode booleans
for col in X.select_dtypes(include="bool").columns:
    X[col] = X[col].astype(int)

# Log-transform target
y_log = np.log1p(y)

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_true = np.expm1(y_test)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── Helper to evaluate a model ────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred_log  = model.predict(X_te)
    pred      = np.expm1(pred_log)
    true      = np.expm1(y_te)
    mape      = mean_absolute_percentage_error(true, pred) * 100
    mae       = mean_absolute_error(true, pred)
    r2        = r2_score(y_te, pred_log)
    print(f"\n{name}")
    print(f"  MAPE : {mape:.1f}%")
    print(f"  MAE  : {mae:.1f} seconds")
    print(f"  R²   : {r2:.3f}")
    return model, mape, r2

# ── Linear Regression ─────────────────────────────────────────────────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LinearRegression())
])
evaluate("Linear Regression", lr_pipe, X_train, y_train, X_test, y_test)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_fitted, xgb_mape, xgb_r2 = evaluate(
    "XGBoost", xgb_model, X_train, y_train, X_test, y_test
)

# ── Cross-validated score for XGBoost ────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, verbosity=0
    ),
    X, y_log, cv=kf, scoring="r2"
)
print(f"\nXGBoost 5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── SHAP feature importance ───────────────────────────────────────────────────
explainer   = shap.TreeExplainer(xgb_fitted)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title("XGBoost Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(r"C:\Users\USER PC\Downloads\demo\shap_importance.png", dpi=150)
plt.show()

# ── Predicted vs Actual plot ──────────────────────────────────────────────────
pred_xgb = np.expm1(xgb_fitted.predict(X_test))
true_vals = np.expm1(y_test)

plt.figure(figsize=(8, 6))
plt.scatter(true_vals, pred_xgb, alpha=0.7, edgecolors="k", linewidths=0.5)
plt.plot([true_vals.min(), true_vals.max()],
         [true_vals.min(), true_vals.max()], "r--", lw=2, label="Perfect prediction")
plt.xlabel("Actual Render Time (seconds)")
plt.ylabel("Predicted Render Time (seconds)")
plt.title("XGBoost: Predicted vs Actual Render Time")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\USER PC\Downloads\demo\predicted_vs_actual.png", dpi=150)
plt.show()

print("\nDone. Check your demo folder for the saved plots.")
