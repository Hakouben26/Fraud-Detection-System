
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
MODEL_PATH = ROOT / "models" / "random_forest.joblib"

THRESHOLD = 0.55


def main():

    print("Loading test data...")

    test_df = pd.read_csv(DATA_PATH)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    print("Loading trained Random Forest model...")

    model = joblib.load(MODEL_PATH)

    print("\nGenerating fraud probabilities...")

    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= THRESHOLD).astype(int)

    print("\n========== FINAL MODEL EVALUATION ==========")

    print(f"Threshold: {THRESHOLD}")

    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_proba):.4f}")

    print("\n========== CONFUSION MATRIX ==========")

    print(confusion_matrix(y_test, y_pred))

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    print("\nFinal evaluation completed!")


if __name__ == "__main__":
    main()
