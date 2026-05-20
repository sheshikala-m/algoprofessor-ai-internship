"""
Time Series Forecasting Agent Pipeline
Day 15 — Extends the Time Series learning track

Workflow:
  1. TimeSeriesLoaderAgent  — load / generate data
  2. PreprocessingAgent     — clean, resample, decompose
  3. ForecastingAgent       — Prophet - > ARIMA - > LSTM cascade
  4. EvaluationAgent        — RMSE / MAE / MAPE metrics
  5. VisualisationAgent     — saves matplotlib chart to outputs/

Dependencies:
  pip install pandas numpy matplotlib scikit-learn statsmodels
  pip install prophet        # optional — falls back to ARIMA
  pip install torch          # optional — falls back gracefully
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


# ─── 1. TimeSeriesLoaderAgent ─────────────────────────────────────────────────

class TimeSeriesLoaderAgent:
    """Load a CSV or generate a synthetic daily time series."""

    name = "TimeSeriesLoaderAgent"

    def load_csv(self, path: str, date_col: str, value_col: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
        df = df.sort_values("ds").reset_index(drop=True)
        print(f"[{self.name}] Loaded {len(df)} rows from {path}")
        return df

    def generate_synthetic(
        self,
        n_days: int = 365,
        trend: float = 0.5,
        seasonality: float = 20.0,
        noise: float = 5.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        np.random.seed(seed)
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
        t     = np.arange(n_days)
        trend_component      = trend * t
        seasonality_weekly   = seasonality * np.sin(2 * np.pi * t / 7)
        seasonality_yearly   = seasonality * 1.5 * np.sin(2 * np.pi * t / 365)
        noise_component      = np.random.normal(0, noise, n_days)
        y = 100 + trend_component + seasonality_weekly + seasonality_yearly + noise_component

        df = pd.DataFrame({"ds": dates, "y": y})
        print(f"[{self.name}] Generated {n_days}-day synthetic series  "
              f"(mean={y.mean():.1f}, std={y.std():.1f})")
        return df


# ─── 2. PreprocessingAgent ───────────────────────────────────────────────────

class PreprocessingAgent:
    name = "PreprocessingAgent"

    def process(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.set_index("ds").resample("D").mean().interpolate().reset_index()
        df["y"] = df["y"].fillna(df["y"].median())

        # Rolling stats
        df["rolling_7"]  = df["y"].rolling(7,  min_periods=1).mean()
        df["rolling_30"] = df["y"].rolling(30, min_periods=1).mean()

        # Train / test split (80/20)
        split = int(len(df) * 0.8)
        train = df.iloc[:split].copy()
        test  = df.iloc[split:].copy()

        print(f"[{self.name}] Preprocessed: train={len(train)}, test={len(test)}")
        return {"df": df, "train": train, "test": test}


# ─── 3. ForecastingAgent ─────────────────────────────────────────────────────

class ForecastingAgent:
    name = "ForecastingAgent"

    # ── ARIMA (always available via statsmodels) ──────────────────────────────

    def _arima_forecast(self, train: pd.DataFrame, horizon: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train["y"].values, order=(2, 1, 2))
        fit   = model.fit()
        fc    = fit.forecast(steps=horizon)
        print(f"[{self.name}] ARIMA(2,1,2) forecast: {horizon} steps")
        return np.array(fc)

    # ── Prophet (optional) ────────────────────────────────────────────────────

    def _prophet_forecast(self, train: pd.DataFrame, horizon: int) -> Optional[np.ndarray]:
        try:
            from prophet import Prophet
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False, interval_width=0.95)
            m.fit(train[["ds", "y"]])
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)
            fc = forecast["yhat"].values[-horizon:]
            print(f"[{self.name}] Prophet forecast: {horizon} steps")
            return fc
        except ImportError:
            print(f"[{self.name}] Prophet not installed — skipping")
            return None

    # ── LSTM (optional) ───────────────────────────────────────────────────────

    def _lstm_forecast(
        self, train: pd.DataFrame, horizon: int, lookback: int = 30
    ) -> Optional[np.ndarray]:
        try:
            import torch
            import torch.nn as nn

            values = train["y"].values.astype(np.float32)
            mean, std = values.mean(), values.std() + 1e-8
            norm = (values - mean) / std

            X, y_t = [], []
            for i in range(lookback, len(norm)):
                X.append(norm[i - lookback: i])
                y_t.append(norm[i])
            X_t = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
            y_t = torch.tensor(np.array(y_t), dtype=torch.float32).unsqueeze(-1)

            class SimpleLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 32, batch_first=True)
                    self.fc   = nn.Linear(32, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            model   = SimpleLSTM()
            opt     = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            for _ in range(30):         # quick training
                opt.zero_grad()
                pred = model(X_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                opt.step()

            # Auto-regressive inference
            window = list(norm[-lookback:])
            preds  = []
            model.eval()
            with torch.no_grad():
                for _ in range(horizon):
                    inp = torch.tensor([window[-lookback:]], dtype=torch.float32).unsqueeze(-1)
                    p   = model(inp).item()
                    preds.append(p)
                    window.append(p)

            fc = np.array(preds) * std + mean
            print(f"[{self.name}] LSTM forecast: {horizon} steps")
            return fc
        except ImportError:
            print(f"[{self.name}] PyTorch not installed — skipping LSTM")
            return None

    # ── Ensemble ─────────────────────────────────────────────────────────────

    def forecast(self, train: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
        results = {}
        arima = self._arima_forecast(train, horizon)
        results["ARIMA"] = arima

        prophet = self._prophet_forecast(train, horizon)
        if prophet is not None:
            results["Prophet"] = prophet

        lstm = self._lstm_forecast(train, horizon)
        if lstm is not None:
            results["LSTM"] = lstm

        # Ensemble = mean of available forecasts
        all_fc = np.stack(list(results.values()))
        results["Ensemble"] = all_fc.mean(axis=0)

        return results


# ─── 4. EvaluationAgent ──────────────────────────────────────────────────────

class EvaluationAgent:
    name = "EvaluationAgent"

    @staticmethod
    def _rmse(actual, pred):
        return float(np.sqrt(np.mean((actual - pred) ** 2)))

    @staticmethod
    def _mae(actual, pred):
        return float(np.mean(np.abs(actual - pred)))

    @staticmethod
    def _mape(actual, pred):
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

    def evaluate(self, test: pd.DataFrame, forecasts: Dict[str, np.ndarray]) -> Dict:
        actual = test["y"].values
        metrics = {}
        for model_name, fc in forecasts.items():
            n = min(len(actual), len(fc))
            a, p = actual[:n], fc[:n]
            metrics[model_name] = {
                "RMSE": round(self._rmse(a, p), 3),
                "MAE":  round(self._mae(a, p), 3),
                "MAPE": round(self._mape(a, p), 3),
            }
            print(f"[{self.name}] {model_name:10s}  "
                  f"RMSE={metrics[model_name]['RMSE']:8.3f}  "
                  f"MAE={metrics[model_name]['MAE']:8.3f}  "
                  f"MAPE={metrics[model_name]['MAPE']:6.2f}%")
        return metrics


# ─── 5. VisualisationAgent ───────────────────────────────────────────────────

class VisualisationAgent:
    name = "VisualisationAgent"

    def plot(
        self,
        df: pd.DataFrame,
        test: pd.DataFrame,
        forecasts: Dict[str, np.ndarray],
        metrics: Dict,
        out_dir: Path,
    ) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0d1117")
        colors = {"ARIMA": "#58a6ff", "Prophet": "#3fb950",
                  "LSTM": "#f78166", "Ensemble": "#e3b341"}

        # ── Top panel: full series + forecasts ───────────────────────────────
        ax = axes[0]
        ax.set_facecolor("#161b22")
        ax.plot(df["ds"], df["y"], color="#8b949e", lw=1, label="Historical", alpha=0.7)
        ax.plot(df["ds"], df["rolling_30"], color="#ffffff", lw=1.5,
                label="30-day MA", alpha=0.9)

        test_dates = test["ds"].values
        for name, fc in forecasts.items():
            n   = min(len(test_dates), len(fc))
            col = colors.get(name, "#c9d1d9")
            ax.plot(test_dates[:n], fc[:n], color=col, lw=2, label=name, linestyle="--")

        ax.axvline(test["ds"].iloc[0], color="#30363d", lw=1, linestyle=":")
        ax.set_title("Time Series Forecast — SidekickBrowser Agent Pipeline",
                     color="#e6edf3", fontsize=14, pad=12)
        ax.set_ylabel("Value", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=9)

        # ── Bottom panel: metric comparison ──────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#161b22")
        models = list(metrics.keys())
        x      = np.arange(len(models))
        rmses  = [metrics[m]["RMSE"] for m in models]
        maes   = [metrics[m]["MAE"]  for m in models]

        w = 0.35
        bars1 = ax2.bar(x - w/2, rmses, w, label="RMSE",
                        color=[colors.get(m, "#58a6ff") for m in models], alpha=0.85)
        bars2 = ax2.bar(x + w/2, maes,  w, label="MAE",
                        color=[colors.get(m, "#58a6ff") for m in models], alpha=0.55)

        ax2.set_xticks(x)
        ax2.set_xticklabels(models, color="#e6edf3")
        ax2.set_ylabel("Error", color="#8b949e")
        ax2.set_title("Model Comparison (RMSE / MAE)", color="#e6edf3", fontsize=12)
        ax2.tick_params(colors="#8b949e")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#30363d")
        ax2.legend(facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#e6edf3", fontsize=9)

        plt.tight_layout()
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"forecast_{ts}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[{self.name}] Chart saved - > {path}")
        return str(path)


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────

def run_pipeline(csv_path: Optional[str] = None, horizon: int = 30):
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*62}")
    print(" Time Series Forecasting Agent Pipeline")
    print(f"{'='*62}\n")

    loader   = TimeSeriesLoaderAgent()
    df       = loader.load_csv(*csv_path.split(",")) if csv_path else \
               loader.generate_synthetic(n_days=365)

    preparer = PreprocessingAgent()
    data     = preparer.process(df)

    forecaster = ForecastingAgent()
    forecasts  = forecaster.forecast(data["train"], horizon)

    evaluator = EvaluationAgent()
    metrics   = evaluator.evaluate(data["test"], forecasts)

    viz      = VisualisationAgent()
    chart    = viz.plot(data["df"], data["test"], forecasts, metrics, out_dir)

    # Save metrics JSON
    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    metric_path = out_dir / f"metrics_{ts}.json"
    metric_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n Pipeline complete")
    print(f"   Chart   - > {chart}")
    print(f"   Metrics - > {metric_path}")
    return {"metrics": metrics, "chart": chart}


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(csv_path=csv, horizon=30)
