
from pathlib import Path

import pandas as pd

from xgboost import XGBClassifier
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

    print("\n========== XGBOOST TRAINING ==========")

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

    print("XGBoost trained successfully!")

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
            y_val,
            predictions,
            labels=[0, 1]
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

    output_path = REPORT_PATH / "xgboost_threshold_comparison.csv"

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
