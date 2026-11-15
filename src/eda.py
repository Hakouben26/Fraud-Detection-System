
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORTS_PATH = ROOT / "reports"

REPORTS_PATH.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found!")

    return pd.read_csv(DATA_PATH)


# ==========================================
# DUPLICATE ANALYSIS
# ==========================================

def analyze_duplicates(df):

    print("\n========== DUPLICATE ANALYSIS ==========")

    duplicate_count = df.duplicated().sum()

    print(f"Total duplicate rows: {duplicate_count}")

    duplicate_mask = df.duplicated(keep=False)

    print("\nClass distribution among rows involved in duplicates:")

    print(
        df.loc[duplicate_mask, "Class"]
        .value_counts()
    )

    print("\nClass distribution among duplicated occurrences:")

    print(
        df.loc[df.duplicated(), "Class"]
        .value_counts()
    )


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

def plot_class_distribution(df):

    plt.figure(figsize=(9, 6))

    ax = sns.countplot(
        data=df,
        x="Class"
    )

    plt.title(
        "Legitimate vs Fraudulent Transactions",
        fontsize=15
    )

    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")

    ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "class_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# TRANSACTION AMOUNT DISTRIBUTION
# ==========================================

def plot_amount_distribution(df):

    plt.figure(figsize=(11, 6))

    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=60,
        log_scale=(False, True),
        element="step",
        stat="count",
        common_norm=False
    )

    plt.title(
        "Transaction Amount Distribution",
        fontsize=15
    )

    plt.xlabel("Transaction Amount")
    plt.ylabel("Count (log scale)")

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "amount_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# TRANSACTIONS OVER TIME
# ==========================================

def plot_transactions_over_time(df):

    plt.figure(figsize=(12, 6))

    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=100,
        element="step",
        stat="count",
        common_norm=False
    )

    plt.title(
        "Transaction Distribution Over Time",
        fontsize=15
    )

    plt.xlabel("Time (seconds since first transaction)")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "transactions_over_time.png",
        dpi=300
    )

    plt.close()


# ==========================================
# CORRELATION HEATMAP
# ==========================================

def plot_correlation_heatmap(df):

    plt.figure(figsize=(18, 14))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.2
    )

    plt.title(
        "Feature Correlation Heatmap",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ==========================================
# FRAUD VS LEGITIMATE STATISTICS
# ==========================================

def analyze_amount_statistics(df):

    print("\n========== AMOUNT ANALYSIS ==========")

    statistics = df.groupby("Class")["Amount"].agg([
        "count",
        "mean",
        "median",
        "min",
        "max",
        "std"
    ])

    print(statistics)

    print("\n========== FRAUD RATE ==========")

    print(
        f"Fraud percentage: {df['Class'].mean() * 100:.4f}%"
    )


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    print("Loading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    analyze_duplicates(df)

    analyze_amount_statistics(df)

    print("\nGenerating visualizations...")

    plot_class_distribution(df)

    plot_amount_distribution(df)

    plot_transactions_over_time(df)

    plot_correlation_heatmap(df)

    print("\nEDA completed successfully!")

    print(f"Charts saved in: {REPORTS_PATH}")


if __name__ == "__main__":
    main()
