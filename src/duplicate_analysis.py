
from pathlib import Path
import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nOriginal dataset shape: {df.shape}")

    # ======================================
    # 1. FULL ROW DUPLICATES
    # ======================================

    print("\n========== FULL ROW DUPLICATES ==========")

    full_duplicates = df.duplicated().sum()

    print(f"Exact duplicate rows: {full_duplicates}")

    # ======================================
    # 2. FEATURE DUPLICATES
    # ======================================

    features = df.columns.drop("Class")

    grouped = (
        df.groupby(list(features), dropna=False)["Class"]
        .agg(["size", "nunique"])
        .reset_index()
    )

    repeated_groups = grouped[grouped["size"] > 1]

    mixed_label_groups = repeated_groups[
        repeated_groups["nunique"] > 1
    ]

    print("\n========== FEATURE DUPLICATES ==========")

    print(f"Repeated feature groups: {len(repeated_groups)}")

    print(
        "Rows belonging to repeated feature groups:",
        repeated_groups["size"].sum()
    )

    # ======================================
    # 3. MIXED LABEL GROUPS
    # ======================================

    print("\n========== MIXED LABEL GROUPS ==========")

    print(
        f"Groups with conflicting labels: {len(mixed_label_groups)}"
    )

    if not mixed_label_groups.empty:

        print("\nExamples of conflicting groups:")

        print(mixed_label_groups.head(10))

    else:

        print("No conflicting labels found.")

    # ======================================
    # 4. DUPLICATE CLASS DISTRIBUTION
    # ======================================

    print("\n========== DUPLICATE CLASS DISTRIBUTION ==========")

    duplicate_mask = df.duplicated(
        subset=list(features),
        keep=False
    )

    duplicate_classes = (
        df.loc[duplicate_mask, "Class"]
        .value_counts()
        .sort_index()
    )

    print(duplicate_classes)

    # ======================================
    # 5. ORIGINAL CLASS DISTRIBUTION
    # ======================================

    print("\n========== ORIGINAL CLASS DISTRIBUTION ==========")

    print(df["Class"].value_counts().sort_index())

    print(
        f"\nOriginal fraud percentage: "
        f"{df['Class'].mean() * 100:.4f}%"
    )

    # ======================================
    # 6. SIMULATE DUPLICATE REMOVAL
    # ======================================

    print("\n========== DATASET AFTER DUPLICATE REMOVAL ==========")

    df_clean = df.drop_duplicates().copy()

    print("Original rows:", len(df))

    print("Rows after removal:", len(df_clean))

    print("Removed rows:", len(df) - len(df_clean))

    print("\nClass distribution after removal:")

    print(df_clean["Class"].value_counts().sort_index())

    print(
        f"\nFraud percentage after removal: "
        f"{df_clean['Class'].mean() * 100:.4f}%"
    )

    # ======================================
    # 7. COMPARISON
    # ======================================

    print("\n========== BEFORE VS AFTER COMPARISON ==========")

    comparison = pd.DataFrame({

        "Before Removal": df["Class"].value_counts(),

        "After Removal": df_clean["Class"].value_counts()

    }).fillna(0).astype(int)

    comparison.index = [
        "Legitimate" if i == 0 else "Fraud"
        for i in comparison.index
    ]

    print(comparison)

    print("\nDuplicate investigation completed!")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()
