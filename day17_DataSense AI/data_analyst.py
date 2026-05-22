"""
data_analyst.py — DataAnalyst Agent
Responsible for: loading data, cleaning, EDA, and feature profiling
Part of DataSense AI multi-agent analytics platform
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


class DataAnalystAgent:
    """
    DataAnalyst Agent
    -----------------
    Primary data ingestion and exploratory analysis agent.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.name = "DataAnalyst"
        self.output_dir = output_dir
        self.df = None
        self.profile = {}

        os.makedirs(output_dir, exist_ok=True)

        print(f"[{self.name}] Agent initialized")

    # ------------------------------------------------------------
    # 1. DATA LOADING
    # ------------------------------------------------------------
    def load_data(self, path: str) -> pd.DataFrame:

        print(f"[{self.name}] Loading data from: {path}")

        ext = os.path.splitext(path)[-1].lower()

        if ext == ".csv":
            self.df = pd.read_csv(path)

        elif ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(path)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        print(
            f"[{self.name}] Loaded {len(self.df)} rows x {len(self.df.columns)} columns"
        )

        return self.df

    # ------------------------------------------------------------
    # 2. DATA CLEANING
    # ------------------------------------------------------------
    def clean_data(self) -> pd.DataFrame:

        print(f"[{self.name}] Starting data cleaning pipeline...")

        original_shape = self.df.shape

        # Remove empty rows
        self.df.dropna(how="all", inplace=True)

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Parse date columns
        for col in self.df.columns:

            if "date" in col.lower() or "time" in col.lower():

                try:
                    self.df[col] = pd.to_datetime(self.df[col])

                    print(f"[{self.name}] Parsed '{col}' as datetime")

                except Exception:
                    pass

        # Fill missing values
        for col in self.df.columns:

            if self.df[col].isnull().sum() > 0:

                if self.df[col].dtype in [np.float64, np.int64]:

                    self.df[col].fillna(
                        self.df[col].median(),
                        inplace=True
                    )

                else:

                    self.df[col].fillna(
                        self.df[col].mode()[0],
                        inplace=True
                    )

        print(
            f"[{self.name}] Cleaning complete: "
            f"{original_shape} -> {self.df.shape}"
        )

        return self.df

    # ------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------
    def run_eda(self) -> dict:

        print(f"[{self.name}] Running Exploratory Data Analysis...")

        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        cat_cols = self.df.select_dtypes(include="object").columns.tolist()

        date_cols = self.df.select_dtypes(include="datetime").columns.tolist()

        self.profile = {

            "shape": list(self.df.shape),

            "columns": self.df.columns.tolist(),

            "dtypes": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },

            "numeric_columns": numeric_cols,

            "categorical_columns": cat_cols,

            "datetime_columns": date_cols,

            "missing_values": self.df.isnull().sum().to_dict(),

            "descriptive_stats": (
                self.df[numeric_cols]
                .describe()
                .round(3)
                .to_dict()
            ),

            "sample_rows": (
                self.df.head(3)
                .to_dict(orient="records")
            ),

            "eda_timestamp": datetime.now().isoformat(),
        }

        print(f"[{self.name}] Numeric columns: {len(numeric_cols)}")
        print(f"[{self.name}] Categorical columns: {len(cat_cols)}")
        print(f"[{self.name}] Date columns: {len(date_cols)}")
        print(
            f"[{self.name}] Missing values: "
            f"{self.df.isnull().sum().sum()}"
        )

        return self.profile

    # ------------------------------------------------------------
    # 4. CORRELATION HEATMAP
    # ------------------------------------------------------------
    def plot_correlation_heatmap(self) -> str:

        numeric_cols = self.df.select_dtypes(include=np.number).columns

        corr = self.df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            linewidths=0.5,
        )

        ax.set_title(
            "Feature Correlation Heatmap",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "correlation_heatmap.png"
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"[{self.name}] Saved heatmap -> {path}")

        return path

    # ------------------------------------------------------------
    # 5. DISTRIBUTION PLOTS
    # ------------------------------------------------------------
    def plot_distributions(self) -> str:

        numeric_cols = (
            self.df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        n = len(numeric_cols)

        cols = 3

        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(16, rows * 3)
        )

        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):

            axes[i].hist(
                self.df[col].dropna(),
                bins=30,
                color="#4A90D9",
                edgecolor="white",
                alpha=0.85,
            )

            axes[i].set_title(
                col,
                fontsize=10,
                fontweight="bold"
            )

            axes[i].grid(alpha=0.3)

        # Hide empty plots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Feature Distributions",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "distributions.png"
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"[{self.name}] Saved distributions -> {path}")

        return path

    # ------------------------------------------------------------
    # 6. EXPORT PROFILE JSON
    # ------------------------------------------------------------
    def export_profile_json(self) -> str:

        path = os.path.join(
            self.output_dir,
            "data_profile.json"
        )

        with open(path, "w") as f:
            json.dump(
                self.profile,
                f,
                indent=2,
                default=str
            )

        print(f"[{self.name}] EDA profile saved -> {path}")

        return path

    # ------------------------------------------------------------
    # 7. AGENT MESSAGE
    # ------------------------------------------------------------
    def get_agent_message(self) -> dict:

        return {

            "from_agent": self.name,

            "status": "complete",

            "data_shape": list(self.df.shape),

            "numeric_cols": (
                self.df.select_dtypes(include=np.number)
                .columns
                .tolist()
            ),

            "cat_cols": (
                self.df.select_dtypes(include="object")
                .columns
                .tolist()
            ),

            "date_cols": (
                self.df.select_dtypes(include="datetime")
                .columns
                .tolist()
            ),

            "profile": self.profile,

            "df_ref": self.df,
        }