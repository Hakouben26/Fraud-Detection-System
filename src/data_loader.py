
from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def load_data():
    """
    Load the credit card transactions dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def inspect_data(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nDataset Information:")
    print(df.info())

    print("\nFirst 5 Rows:")
    print(df.head())

    if "Class" in df.columns:
        print("\nClass Distribution:")
        print(df["Class"].value_counts())

        print("\nClass Percentages:")
        print(
            df["Class"]
            .value_counts(normalize=True)
            .mul(100)
            .round(4)
        )


if __name__ == "__main__":

    data = load_data()

    inspect_data(data)
