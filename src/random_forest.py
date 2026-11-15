
from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed"

MODEL_PATH = ROOT / "models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

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


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

def train_model(X_train, y_train):

    print("\n========== RANDOM FOREST TRAINING ==========")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Random Forest trained successfully!")

    return model


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test):

    print("\n========== MODEL EVALUATION ==========")

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_proba
    )

    pr_auc = average_precision_score(
        y_test, y_proba
    )

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


# ==========================================
# SAVE MODEL
# ==========================================

def save_model(model):

    output_path = MODEL_PATH / "random_forest.joblib"

    joblib.dump(model, output_path)

    print("\nModel saved successfully!")
    print(f"Location: {output_path}")


# ==========================================
# MAIN
# ==========================================

def main():

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\n===================================")
    print("RANDOM FOREST COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()
