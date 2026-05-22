"""
statistics_engine.py — StatisticsEngine Agent
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from scipy import stats


class StatisticsEngineAgent:

    def __init__(self, output_dir="outputs"):

        self.name = "StatisticsEngine"
        self.output_dir = output_dir
        self.df = None
        self.stats_report = {}

        os.makedirs(output_dir, exist_ok=True)

        print(f"[{self.name}] Agent initialized")

    # =========================================================
    # RECEIVE DATA
    # =========================================================
    def receive_data(self, agent_message):

        self.df = agent_message["df_ref"]
        self.numeric_cols = agent_message["numeric_cols"]
        self.cat_cols = agent_message["cat_cols"]

        print(
            f"[{self.name}] Received data from "
            f"{agent_message['from_agent']} "
            f"({self.df.shape[0]} rows, {self.df.shape[1]} cols)"
        )

    # =========================================================
    # NORMALITY TESTS
    # =========================================================
    def normality_tests(self):

        print(f"[{self.name}] Running normality tests...")

        results = {}

        sample = self.df[self.numeric_cols].sample(
            min(500, len(self.df)),
            random_state=42
        )

        for col in self.numeric_cols:

            stat, p = stats.shapiro(
                sample[col].dropna()
            )

            results[col] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "is_normal": bool(p > 0.05),
            }

            status = "NORMAL" if p > 0.05 else "NON-NORMAL"

            print(
                f"[{self.name}] {col}: "
                f"p={p:.4f} | {status}"
            )

        self.stats_report["normality"] = results

        return results

    # =========================================================
    # OUTLIER DETECTION
    # =========================================================
    def detect_outliers(self):

        print(f"[{self.name}] Detecting outliers...")

        outlier_summary = {}

        for col in self.numeric_cols:

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[col] < lower) |
                (self.df[col] > upper)
            ]

            pct = round(
                len(outliers) / len(self.df) * 100,
                2
            )

            outlier_summary[col] = {
                "count": len(outliers),
                "percentage": pct,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }

            if pct > 5:
                print(
                    f"[{self.name}] Warning: "
                    f"{col} has {pct}% outliers"
                )

        self.stats_report["outliers"] = outlier_summary

        return outlier_summary

    # =========================================================
    # CORRELATION SIGNIFICANCE
    # =========================================================
    def correlation_significance(self):

        print(f"[{self.name}] Computing correlation significance...")

        sig_pairs = []

        cols = self.numeric_cols

        for i in range(len(cols)):

            for j in range(i + 1, len(cols)):

                r, p = stats.pearsonr(
                    self.df[cols[i]].dropna(),
                    self.df[cols[j]].dropna(),
                )

                if p < 0.05:

                    sig_pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "pearson_r": round(float(r), 4),
                        "p_value": round(float(p), 6),
                        "significant": True,
                    })

        print(
            f"[{self.name}] Found "
            f"{len(sig_pairs)} significant correlations"
        )

        self.stats_report["significant_correlations"] = sig_pairs

        return sig_pairs

    # =========================================================
    # GROUP STATISTICS
    # =========================================================
    def group_statistics(
        self,
        group_col=None,
        value_col="sales_amount"
    ):

        if group_col is None or group_col not in self.df.columns:

            group_col = (
                self.cat_cols[0]
                if self.cat_cols else None
            )

        if group_col is None:
            return {}

        print(
            f"[{self.name}] Group statistics: "
            f"{value_col} by {group_col}"
        )

        grouped = (
            self.df.groupby(group_col)[value_col]
            .agg(["mean", "std", "count", "min", "max"])
            .round(3)
            .reset_index()
        )

        grouped.columns = [
            group_col,
            "mean",
            "std",
            "count",
            "min",
            "max"
        ]

        result = grouped.to_dict(orient="records")

        self.stats_report["group_stats"] = {
            "group_col": group_col,
            "value_col": value_col,
            "data": result,
        }

        return result

    # =========================================================
    # BOXPLOT
    # =========================================================
    def plot_boxplots(self):

        cols = self.numeric_cols[:6]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 8)
        )

        axes = axes.flatten()

        for i, col in enumerate(cols):

            axes[i].boxplot(
                self.df[col].dropna(),
                vert=True,
                patch_artist=True,
            )

            axes[i].set_title(col)
            axes[i].grid(axis="y", alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Boxplot Analysis")

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "boxplots.png"
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"[{self.name}] Boxplot saved: {path}")

        return path

    # =========================================================
    # EXPORT JSON
    # =========================================================
    def export_stats_json(self):

        path = os.path.join(
            self.output_dir,
            "statistics_report.json"
        )

        with open(path, "w") as f:

            json.dump(
                self.stats_report,
                f,
                indent=2,
                default=str
            )

        print(f"[{self.name}] Statistics report saved: {path}")

        return path

    # =========================================================
    # AGENT MESSAGE
    # =========================================================
    def get_agent_message(self):

        return {
            "from_agent": self.name,
            "status": "complete",
            "stats_report": self.stats_report,
            "df_ref": self.df,
            "numeric_cols": self.numeric_cols,
            "cat_cols": self.cat_cols,
        }