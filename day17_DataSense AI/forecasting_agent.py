"""
forecasting_agent.py — ForecastingAgent
Responsible for: time series analysis, trend detection, 30-day sales forecasting
Part of DataSense AI multi-agent analytics platform
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ForecastingAgent:
    """
    ForecastingAgent
    ----------------
    Performs time series decomposition and forecasting.
    Uses polynomial regression for trend + seasonality modeling.
    Generates PNG forecast charts as deliverable outputs.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.name = "ForecastingAgent"
        self.output_dir = output_dir
        self.df = None
        self.forecast_results = {}
        os.makedirs(output_dir, exist_ok=True)
        print(f"[{self.name}] Agent initialized")

    def receive_data(self, df: pd.DataFrame, date_col: str = "date", value_col: str = "sales_amount"):
        """Load time series data."""
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col

        if date_col in df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col)

        print(f"[{self.name}] Time series data received: {len(self.df)} records")

    # ---------------------------------------------------------
    # 1. AGGREGATE TIME SERIES
    # ---------------------------------------------------------
    def aggregate_daily(self) -> pd.DataFrame:
        """Aggregate data to daily granularity."""
        print(f"[{self.name}] Aggregating to daily time series...")

        self.ts = (
            self.df.groupby(self.date_col)[self.value_col]
            .sum()
            .reset_index()
            .rename(columns={self.date_col: "date", self.value_col: "value"})
        )

        self.ts["day_index"] = np.arange(len(self.ts))

        print(f"[{self.name}] Time series length: {len(self.ts)} days")

        return self.ts

    # ---------------------------------------------------------
    # 2. ROLLING STATISTICS
    # ---------------------------------------------------------
    def compute_rolling_stats(self, window: int = 7) -> pd.DataFrame:
        """Add rolling mean and std to time series."""

        self.ts[f"rolling_mean_{window}d"] = (
            self.ts["value"].rolling(window).mean()
        )

        self.ts[f"rolling_std_{window}d"] = (
            self.ts["value"].rolling(window).std()
        )

        print(f"[{self.name}] Rolling stats computed (window={window})")

        return self.ts

    # ---------------------------------------------------------
    # 3. POLYNOMIAL FORECAST
    # ---------------------------------------------------------
    def forecast_polynomial(self, forecast_days: int = 30, degree: int = 3) -> dict:
        """Fit polynomial regression and forecast future values."""

        print(f"[{self.name}] Forecasting {forecast_days} days ahead (degree={degree})...")

        X = self.ts[["day_index"]].values
        y = self.ts["value"].values

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Historical predictions
        y_pred = model.predict(X_poly)

        # Future forecast
        future_idx = np.arange(
            len(self.ts),
            len(self.ts) + forecast_days
        ).reshape(-1, 1)

        future_poly = poly.transform(future_idx)
        future_pred = model.predict(future_poly)

        # Prevent negative forecasts
        future_pred = np.clip(future_pred, 0, None)

        # Future dates
        last_date = self.ts["date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast": future_pred.round(2)
        })

        self.forecast_df = forecast_df
        self.y_pred_hist = y_pred

        results = {
            "model": "Polynomial Regression",
            "degree": degree,
            "forecast_days": forecast_days,
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "forecast_summary": {
                "min": round(float(future_pred.min()), 2),
                "max": round(float(future_pred.max()), 2),
                "mean": round(float(future_pred.mean()), 2),
                "total": round(float(future_pred.sum()), 2),
            },
            "forecast_data": forecast_df.to_dict(orient="records"),
        }

        self.forecast_results = results

        print(f"[{self.name}] MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        print(f"[{self.name}] 30-day forecast total: {future_pred.sum():.2f}")

        return results

    # ---------------------------------------------------------
    # 4. SIMPLE MOVING AVERAGE
    # ---------------------------------------------------------
    def forecast_sma(self, window: int = 7) -> dict:
        """Simple Moving Average baseline forecast."""

        sma_value = self.ts["value"].tail(window).mean()

        print(f"[{self.name}] SMA({window}) baseline forecast: {sma_value:.2f}/day")

        self.forecast_results["sma_baseline"] = {
            "window": window,
            "daily_avg_forecast": round(float(sma_value), 2),
            "30_day_total": round(float(sma_value * 30), 2),
        }

        return self.forecast_results["sma_baseline"]

    # ---------------------------------------------------------
    # 5. FORECAST CHART
    # ---------------------------------------------------------
    def plot_forecast(self) -> str:
        """Generate forecast visualization."""

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        fig.patch.set_facecolor("#0d1117")

        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")

            ax.spines["bottom"].set_color("#30363d")
            ax.spines["left"].set_color("#30363d")
            ax.spines["top"].set_color("#30363d")
            ax.spines["right"].set_color("#30363d")

            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        # Top chart
        ax1 = axes[0]

        ax1.plot(
            self.ts["date"],
            self.ts["value"],
            color="#58a6ff",
            linewidth=1.5,
            alpha=0.8,
            label="Actual Sales"
        )

        ax1.plot(
            self.ts["date"],
            self.y_pred_hist,
            color="#f78166",
            linewidth=2,
            linestyle="--",
            label="Fitted Trend"
        )

        if hasattr(self, "forecast_df"):

            ax1.plot(
                self.forecast_df["date"],
                self.forecast_df["forecast"],
                color="#3fb950",
                linewidth=2.5,
                linestyle="-",
                label="30-Day Forecast"
            )

            fc = self.forecast_df["forecast"].values

            ax1.fill_between(
                self.forecast_df["date"],
                fc * 0.9,
                fc * 1.1,
                color="#3fb950",
                alpha=0.15,
                label="+/-10% Confidence Band"
            )

            ax1.axvline(
                x=self.ts["date"].max(),
                color="#e3b341",
                linewidth=1.5,
                linestyle=":",
                label="Forecast Start"
            )

        ax1.set_title(
            "Sales Forecast - Polynomial Regression Model",
            fontsize=13,
            fontweight="bold",
            pad=10
        )

        ax1.set_ylabel("Sales Amount ($)")

        ax1.legend(
            loc="upper left",
            facecolor="#21262d",
            edgecolor="#30363d",
            labelcolor="white",
            fontsize=9
        )

        ax1.grid(alpha=0.15, color="#30363d")

        # Bottom chart
        ax2 = axes[1]

        ax2.plot(
            self.ts["date"],
            self.ts["value"],
            color="#58a6ff",
            alpha=0.4,
            linewidth=1,
            label="Daily Sales"
        )

        if "rolling_mean_7d" in self.ts.columns:
            ax2.plot(
                self.ts["date"],
                self.ts["rolling_mean_7d"],
                color="#bc8cff",
                linewidth=2,
                label="7-Day Rolling Mean"
            )

        if "rolling_std_7d" in self.ts.columns:

            mean = self.ts["rolling_mean_7d"]
            std = self.ts["rolling_std_7d"]

            ax2.fill_between(
                self.ts["date"],
                mean - std,
                mean + std,
                color="#bc8cff",
                alpha=0.15,
                label="+/-1 Std Dev"
            )

        ax2.set_title(
            "Rolling Average Analysis",
            fontsize=13,
            fontweight="bold",
            pad=10
        )

        ax2.set_ylabel("Sales Amount ($)")
        ax2.set_xlabel("Date")

        ax2.legend(
            loc="upper left",
            facecolor="#21262d",
            edgecolor="#30363d",
            labelcolor="white",
            fontsize=9
        )

        ax2.grid(alpha=0.15, color="#30363d")

        plt.tight_layout(pad=2.0)

        path = os.path.join(self.output_dir, "forecast_chart.png")

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

        plt.close()

        print(f"[{self.name}] Forecast chart saved -> {path}")

        return path

    # ---------------------------------------------------------
    # 6. EXPORT FORECAST JSON
    # ---------------------------------------------------------
    def export_forecast_json(self) -> str:
        """Save forecast results to JSON."""

        path = os.path.join(self.output_dir, "forecast_results.json")

        with open(path, "w") as f:
            json.dump(self.forecast_results, f, indent=2, default=str)

        print(f"[{self.name}] Forecast JSON saved -> {path}")

        return path