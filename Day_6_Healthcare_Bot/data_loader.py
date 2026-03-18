# ============================================================
# DATA LOADER
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Titanic dataset loading and summarization for Day 6
# ============================================================

# This file provides data loading and summarization functions
# used by all other Day 6 files. It is self-contained so Day 6
# runs independently without needing Day 5 files in the path.
# The Titanic dataset is loaded from seaborn if available,
# otherwise an inline 20-row sample is used automatically.

import pandas as pd
import numpy as np

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ============================================================
# DATA LOADING
# ============================================================

def load_titanic():
    """
    Loads the Titanic dataset from seaborn if available.
    Falls back to a representative 20-row inline sample
    so all Day 6 files run without seaborn installed.
    Prints confirmation of which source was used.
    """
    if SEABORN_AVAILABLE:
        df = sns.load_dataset("titanic")
        print("Loaded Titanic dataset from seaborn: " +
              str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns.")
        return df

    data = {
        "survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        "pclass"  : [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3],
        "sex"     : ["male", "female", "female", "female", "male", "male",
                     "male", "male", "female", "female", "female", "female",
                     "male", "male", "male", "female", "male", "male", "female", "female"],
        "age"     : [22, 38, 26, 35, 35, None, 54, 2, 27, 14,
                     4, 58, 20, 39, 14, 55, 2, None, 31, None],
        "fare"    : [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.07,
                     11.13, 30.07, 16.70, 26.55, 8.05, 31.27, 7.85, 16.00,
                     29.12, 13.00, 18.00, 7.22],
        "embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C",
                     "S", "S", "S", "S", "S", "S", "Q", "S", "S", "S"]
    }
    df = pd.DataFrame(data)
    print("Loaded Titanic inline fallback: " +
          str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns.")
    return df


# ============================================================
# DATA SUMMARIZER
# ============================================================

def summarize_dataframe(df):
    """
    Converts a dataframe into a compact text summary suitable
    for sending to an LLM as context. Includes shape, column
    names, missing value counts, numeric statistics, and
    Titanic-specific survival breakdowns where available.
    Returns a multi-line string ready to paste into a prompt.
    """
    lines = []

    lines.append("Dataset shape: " + str(df.shape[0]) +
                 " rows x " + str(df.shape[1]) + " columns")
    lines.append("Columns: " + ", ".join(df.columns.tolist()))
    lines.append("")

    missing = {col: int(df[col].isnull().sum())
               for col in df.columns if df[col].isnull().sum() > 0}
    if missing:
        lines.append("Missing values:")
        for col, count in missing.items():
            pct = round(count / len(df) * 100, 1)
            lines.append("  " + col + ": " + str(count) +
                         " missing (" + str(pct) + "%)")
        lines.append("")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        lines.append("Numeric statistics:")
        for col in num_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                lines.append(
                    "  " + col + ":"
                    " mean=" + str(round(col_data.mean(), 3)) +
                    ", median=" + str(round(col_data.median(), 3)) +
                    ", std=" + str(round(col_data.std(), 3)) +
                    ", min=" + str(round(col_data.min(), 3)) +
                    ", max=" + str(round(col_data.max(), 3))
                )
        lines.append("")

    if "survived" in df.columns:
        overall = round(df["survived"].mean() * 100, 1)
        lines.append("Survival statistics:")
        lines.append("  Overall survival rate: " + str(overall) + "%")

        if "sex" in df.columns:
            for gender in ["male", "female"]:
                subset = df[df["sex"] == gender]["survived"]
                if len(subset) > 0:
                    rate = round(subset.mean() * 100, 1)
                    lines.append("  " + gender.capitalize() +
                                 " survival rate: " + str(rate) + "%")

        if "pclass" in df.columns:
            for cls in [1, 2, 3]:
                subset = df[df["pclass"] == cls]["survived"]
                if len(subset) > 0:
                    rate = round(subset.mean() * 100, 1)
                    lines.append("  Class " + str(cls) +
                                 " survival rate: " + str(rate) + "%")

    return "\n".join(lines)


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("DATA LOADER DEMO")
    print("=" * 55)

    df = load_titanic()

    print("\n-- Dataset Preview (first 5 rows) --")
    preview_cols = [c for c in ["survived", "pclass", "sex", "age", "fare"]
                    if c in df.columns]
    print(df[preview_cols].head(5).to_string())

    print("\n-- Dataset Summary --")
    summary = summarize_dataframe(df)
    print(summary)

    print("\n-- Data Loader demo complete --")


if __name__ == "__main__":
    run_demo()
