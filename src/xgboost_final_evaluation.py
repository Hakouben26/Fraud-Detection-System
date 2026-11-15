
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "xgboost.joblib"
REPORT_PATH = ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.70


def main():

    print("Loading test data...")

    test_df = pd.read_csv(DATA_PATH)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    print("Loading trained XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("\nGenerating predictions...")

    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n========== FINAL XGBOOST EVALUATION ==========")

    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\n========== CONFUSION MATRIX ==========")

    matrix = confusion_matrix(y_test, y_pred)

    print(matrix)

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    results = pd.DataFrame([{
        "Model": "XGBoost",
        "Threshold": THRESHOLD,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }])

    output_path = REPORT_PATH / "xgboost_final_results.csv"

    results.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)

    print("\nXGBOOST FINAL EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
