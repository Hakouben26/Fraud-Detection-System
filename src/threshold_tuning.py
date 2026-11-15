
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
REPORT_PATH = ROOT / "reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    print("Loading training data...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def train_model(X_train, y_train):
    print("\nTraining Random Forest...")

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

    print("Model trained successfully!")

    return model


def evaluate_thresholds(model, X_val, y_val):
    print("\n========== THRESHOLD OPTIMIZATION ==========")

    probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp
        })

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))

    output_path = REPORT_PATH / "threshold_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)

    best = results_df.loc[
        results_df["F1-Score"].idxmax()
    ]

    print("\n========== BEST F1 THRESHOLD ==========")
    print(best.to_string())


def main():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Validation fraud cases:", int(y_val.sum()))

    model = train_model(X_train, y_train)

    evaluate_thresholds(model, X_val, y_val)


if __name__ == "__main__":
    main()
