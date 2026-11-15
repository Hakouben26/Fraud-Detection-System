
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
MODEL_PATH = "models/xgboost.joblib"
TEST_DATA_PATH = "data/processed/test.csv"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 50)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 50)

# 1. Load trained XGBoost model
print("\nLoading XGBoost model...")
model = joblib.load(MODEL_PATH)

# 2. Load test dataset
print("Loading test dataset...")
test_df = pd.read_csv(TEST_DATA_PATH)

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# 3. Use a sample to make SHAP analysis faster
# Stratified sampling preserves the fraud/legitimate ratio
X_sample = X_test.sample(
    n=min(2000, len(X_test)),
    random_state=42
)

print(f"\nSample size: {len(X_sample)}")
print("Calculating SHAP values...")

# 4. Create SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

# 5. Calculate SHAP values
shap_values = explainer(X_sample)

print("SHAP values calculated successfully!")

# 6. Summary plot (beeswarm)
print("\nGenerating SHAP Summary Plot...")

plt.figure(figsize=(12, 8))

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.title(
    "SHAP Summary Plot - XGBoost Fraud Detection",
    fontsize=14
)

plt.tight_layout()

summary_path = os.path.join(
    REPORTS_DIR,
    "shap_summary.png"
)

plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()

# 7. SHAP bar plot
print("Generating SHAP Bar Plot...")

plt.figure(figsize=(12, 8))

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title(
    "SHAP Feature Importance - XGBoost",
    fontsize=14
)

plt.tight_layout()

bar_path = os.path.join(
    REPORTS_DIR,
    "shap_bar.png"
)

plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

# 8. Save mean absolute SHAP values
print("Saving SHAP importance table...")

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean Absolute SHAP": abs(shap_values.values).mean(axis=0)
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

importance_df["Importance (%)"] = (
    importance_df["Mean Absolute SHAP"]
    / importance_df["Mean Absolute SHAP"].sum()
) * 100

csv_path = os.path.join(
    REPORTS_DIR,
    "shap_importance.csv"
)

importance_df.to_csv(csv_path, index=False)

# 9. Display results
print("\n========== TOP 15 SHAP FEATURES ==========")
print(importance_df.head(15).to_string(index=False))

print("\n========== FILES SAVED ==========")
print(f"Summary plot: {summary_path}")
print(f"Bar plot: {bar_path}")
print(f"Importance CSV: {csv_path}")

print("\nSHAP analysis completed successfully!")
