"""
data_preprocessing.py
- Load and inspect the dataset
- Check for missing values
- Show data types and basic statistics
- Visualize distributions and correlations (optional, for script use)
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
def set_plot_style():
    sns.set(style="whitegrid")


def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df


def data_overview(df):
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nInfo:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())


def save_plot(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def check_missing_values(df, reports_dir: Path):
    print("\nMissing values per column:")
    print(df.isnull().sum())
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    save_plot(fig, reports_dir / "missing_values_heatmap.png")


def data_types_and_stats(df):
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe(include='all'))


def save_correlation_heatmap(df, reports_dir: Path):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_plot(fig, reports_dir / "correlation_heatmap.png")


def save_delay_distribution(df, reports_dir: Path):
    fig = plt.figure(figsize=(6, 4))
    sns.histplot(df["delay_days"].dropna(), kde=True)
    plt.title("Delay Days Distribution")
    save_plot(fig, reports_dir / "delay_distribution.png")


def save_delay_vs_complexity(df, reports_dir: Path):
    fig = plt.figure(figsize=(6, 4))
    sns.scatterplot(x="feature_complexity", y="delay_days", data=df)
    plt.title("Delay vs Feature Complexity")
    save_plot(fig, reports_dir / "delay_vs_complexity.png")


def save_delay_vs_blockers(df, reports_dir: Path):
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(x="num_blockers", y="delay_days", data=df)
    plt.title("Delay vs Blockers")
    save_plot(fig, reports_dir / "delay_vs_blockers.png")


def print_eda_insights(df):
    if "delay_days" not in df.columns:
        print("\nEDA Insights: 'delay_days' column not found. Skipping correlation summary.")
        return

    corr = df.corr(numeric_only=True)["delay_days"].drop("delay_days")
    top_pos = corr.sort_values(ascending=False).head(3)
    top_neg = corr.sort_values(ascending=True).head(3)

    print("\nEDA Insights (Correlation with delay_days):")
    print("Top positive drivers:")
    for name, value in top_pos.items():
        print(f"  - {name}: {value:.3f}")

    print("Top negative drivers:")
    for name, value in top_neg.items():
        print(f"  - {name}: {value:.3f}")


def main():
    set_plot_style()
    project_root = Path(__file__).resolve().parents[1]
    file_path = project_root / "data" / "dataset.csv"
    reports_dir = project_root / "reports"
    df = load_data(file_path)
    data_overview(df)
    check_missing_values(df, reports_dir)
    data_types_and_stats(df)
    print_eda_insights(df)
    save_correlation_heatmap(df, reports_dir)
    if "delay_days" in df.columns:
        save_delay_distribution(df, reports_dir)
        save_delay_vs_complexity(df, reports_dir)
        save_delay_vs_blockers(df, reports_dir)

if __name__ == "__main__":
    main()
