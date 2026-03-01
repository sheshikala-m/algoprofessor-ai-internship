"""
Day 04 — Iris Data Engineering & Analysis Pipeline
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To simulate real-world data engineering workflow by loading Iris dataset
from CSV, performing data validation, statistical analysis, correlation
study, and grouped aggregations.
"""

import pandas as pd


def load_dataset():
    """Load dataset from CSV."""
    path = "iris_dataset.csv"
    df = pd.read_csv(path)

    print("\nDataset loaded successfully.")
    return df


def inspect_data(df):
    """Inspect dataset structure."""
    print("\n=== First 5 Records ===")
    print(df.head())

    print("\n=== Dataset Information ===")
    print(df.info())


def check_missing(df):
    """Check and handle missing values."""
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing)


def statistical_analysis(df):
    """Generate statistical summary."""
    print("\n=== Statistical Summary ===")
    stats = df.describe()
    print(stats)


def correlation_analysis(df):
    """Perform correlation study."""
    print("\n=== Correlation Analysis ===")

    correlation = df.corr(numeric_only=True)
    print(correlation)


def grouping_analysis(df):
    """Group analysis by target class."""
    print("\n=== Average Feature Values by Target ===")

    grouped = df.groupby("target").mean()
    print(grouped)


def main():
    print("Starting Iris Data Engineering Pipeline...\n")

    df = load_dataset()
    inspect_data(df)
    check_missing(df)
    statistical_analysis(df)
    correlation_analysis(df)
    grouping_analysis(df)

    print("\nPipeline Execution Completed Successfully.")


if __name__ == "__main__":
    main()
