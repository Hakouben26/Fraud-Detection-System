
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

PROCESSED_PATH = ROOT / "data" / "processed"

MODEL_PATH = ROOT / "models"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    return df


# ==========================================
# REMOVE EXACT DUPLICATES
# ==========================================

def remove_duplicates(df):

    print("\n========== DUPLICATE REMOVAL ==========")

    original_rows = len(df)

    df = df.drop_duplicates().copy()

    removed_rows = original_rows - len(df)

    print(f"Original rows: {original_rows}")
    print(f"Rows after removal: {len(df)}")
    print(f"Removed rows: {removed_rows}")

    print("\nClass distribution after removal:")

    print(df["Class"].value_counts().sort_index())

    print(
        f"\nFraud rate: {df['Class'].mean() * 100:.4f}%"
    )

    return df


# ==========================================
# SPLIT FEATURES AND TARGET
# ==========================================

def split_features_target(df):

    X = df.drop(columns=["Class"])

    y = df["Class"]

    return X, y


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

def split_data(X, y):

    print("\n========== TRAIN / TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining class distribution:")

    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")

    print(y_test.value_counts().sort_index())

    print(
        f"\nTraining fraud rate: {y_train.mean() * 100:.4f}%"
    )

    print(
        f"Testing fraud rate: {y_test.mean() * 100:.4f}%"
    )

    return X_train, X_test, y_train, y_test


# ==========================================
# FEATURE SCALING
# ==========================================

def scale_features(X_train, X_test):

    print("\n========== FEATURE SCALING ==========")

    scaler = StandardScaler()

    columns_to_scale = ["Time", "Amount"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    # Apply the same transformation to test data
    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    print("Scaled columns:", columns_to_scale)

    print("Scaler fitted on training data only.")

    return X_train, X_test, scaler


# ==========================================
# SAVE PROCESSED DATA
# ==========================================

def save_data(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
):

    print("\n========== SAVING PROCESSED DATA ==========")

    train_df = X_train.copy()
    train_df["Class"] = y_train

    test_df = X_test.copy()
    test_df["Class"] = y_test

    train_df.to_csv(
        PROCESSED_PATH / "train.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_PATH / "test.csv",
        index=False
    )

    joblib.dump(
        scaler,
        MODEL_PATH / "scaler.joblib"
    )

    print("Saved: data/processed/train.csv")
    print("Saved: data/processed/test.csv")
    print("Saved: models/scaler.joblib")


# ==========================================
# MAIN
# ==========================================

def main():

    # 1. Load
    df = load_data()

    # 2. Remove exact duplicates
    df = remove_duplicates(df)

    # 3. Separate features and target
    X, y = split_features_target(df)

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Scale features
    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    # 6. Save files
    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )

    print("\n===================================")
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("===================================")


if __name__ == "__main__":
    main()
