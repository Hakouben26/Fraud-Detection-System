
from pathlib import Path

import pandas as pd
import joblib

from xgboost import XGBClassifier

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

DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)


def load_data():

    print("Loading datasets...")

    train_df = pd.read_csv(DATA_PATH / "train.csv")
    test_df = pd.read_csv(DATA_PATH / "test.csv")

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    print("\n========== XGBOOST TRAINING ==========")

    # Handle class imbalance
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print("Negative samples:", negative)
    print("Positive samples:", positive)
    print("Scale Pos Weight:", round(scale_pos_weight, 2))

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    print("\nXGBoost trained successfully!")

    return model


def evaluate_model(model, X_test, y_test):

    print("\n========== MODEL EVALUATION ==========")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

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


def save_model(model):

    output_path = MODEL_PATH / "xgboost.joblib"

    joblib.dump(model, output_path)

    print("\nModel saved successfully!")
    print("Location:", output_path)


def main():

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\n===================================")
    print("XGBOOST TRAINING COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()
