"""
Day 04 — Iris CSV Analysis Pipeline
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To load Iris dataset from CSV file and perform
structured exploratory data analysis.
"""

import pandas as pd


def load_dataset():
    df = pd.read_csv("day01_python_ai_fundamentals/iris_dataset.csv")
    return df


def inspect_data(df):
    print("\n===== First 5 Rows =====")
    print(df.head())

    print("\n===== Dataset Info =====")
    print(df.info())


def statistical_summary(df):
    print("\n===== Statistical Summary =====")
    print(df.describe())


def null_check(df):
    print("\n===== Missing Values =====")
    print(df.isnull().sum())


def main():
    print("Starting Iris CSV EDA Pipeline...\n")

    df = load_dataset()
    inspect_data(df)
    statistical_summary(df)
    null_check(df)

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
