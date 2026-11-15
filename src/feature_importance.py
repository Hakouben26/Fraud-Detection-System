
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "xgboost.joblib"
REPORT_PATH = ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("Extracting feature importance...")

    booster = model.get_booster()

    importance_scores = booster.get_score(
        importance_type="gain"
    )

    all_features = model.get_booster().feature_names

    importance_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": [
            importance_scores.get(feature, 0)
            for feature in all_features
        ]
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df["Importance (%)"] = (
        importance_df["Importance"]
        / importance_df["Importance"].sum()
    ) * 100

    output_csv = REPORT_PATH / "feature_importance.csv"

    importance_df.to_csv(output_csv, index=False)

    print("\n========== TOP 15 FEATURES ==========")

    print(
        importance_df.head(15).to_string(index=False)
    )

    top_features = importance_df.head(15).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title(
        "XGBoost - Top 15 Feature Importance (Gain)",
        fontsize=14
    )

    plt.xlabel("Importance (Gain)")
    plt.ylabel("Features")

    plt.tight_layout()

    output_image = REPORT_PATH / "feature_importance.png"

    plt.savefig(output_image, dpi=300)

    plt.close()

    print("\nFeature importance saved successfully!")
    print("CSV:", output_csv)
    print("Chart:", output_image)


if __name__ == "__main__":
    main()
