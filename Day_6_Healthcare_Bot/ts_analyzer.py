# ============================================================
# TIME SERIES ANALYZER
# Day 6: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Time series analysis with LLM-generated insights
# ============================================================

# Time series data has a time component which makes it
# different from regular tabular data. Patterns like trends,
# seasonality, and anomalies are important to detect.
# This file generates synthetic time series data representing
# ML experiment metrics over time and uses an LLM to generate
# natural language insights about the patterns found.
# The analysis combines statistical methods with LLM narration
# to produce human-readable reports about time series trends.

import math
import random
import statistics
from datetime import datetime, timedelta
from ollama_setup import OllamaClient

MODEL = "llama3"


# ============================================================
# SYNTHETIC TIME SERIES GENERATOR
# ============================================================

def generate_experiment_metrics(days=30, seed=42):
    """
    Generates synthetic daily ML experiment metrics over time.
    Simulates a realistic training scenario where accuracy
    improves over time with some noise and occasional drops.

    Parameters:
        days : number of days to generate data for
        seed : random seed for reproducibility

    Returns:
        list of dicts with date, accuracy, loss, f1_score
    """
    random.seed(seed)
    records = []
    base_accuracy = 0.70
    base_loss     = 0.80

    start_date = datetime(2026, 1, 1)

    for i in range(days):
        date = start_date + timedelta(days=i)

        # accuracy improves over time with noise
        trend     = i * 0.007
        noise     = random.uniform(-0.02, 0.02)
        accuracy  = min(0.98, base_accuracy + trend + noise)

        # loss decreases over time with noise
        loss_trend = i * 0.008
        loss_noise = random.uniform(-0.03, 0.03)
        loss       = max(0.05, base_loss - loss_trend + loss_noise)

        # f1 follows accuracy with slight variation
        f1_noise = random.uniform(-0.015, 0.015)
        f1_score = min(0.97, accuracy - 0.02 + f1_noise)

        records.append({
            "date"    : date.strftime("%Y-%m-%d"),
            "day"     : i + 1,
            "accuracy": round(accuracy, 4),
            "loss"    : round(loss, 4),
            "f1_score": round(f1_score, 4)
        })

    return records


def generate_experiment_runs(num_experiments=8, seed=42):
    """
    Generates synthetic data for multiple experiment runs
    representing different model configurations tested over time.

    Returns:
        list of dicts with experiment details and final metrics
    """
    random.seed(seed)
    models    = ["BERT", "RoBERTa", "DistilBERT", "GPT-2",
                 "ResNet50", "EfficientNet", "LSTM", "Transformer"]
    datasets  = ["NLP-Corpus-v2", "SentimentData-v1", "ImageNet-Subset",
                 "TimeSeriesData-v3"]
    researchers = ["Ananya", "Vikram", "Priya", "Rohan"]

    experiments = []
    start_date  = datetime(2026, 1, 1)

    for i in range(num_experiments):
        date     = start_date + timedelta(days=i * 3)
        accuracy = round(random.uniform(0.78, 0.96), 4)
        f1       = round(accuracy - random.uniform(0.01, 0.04), 4)
        loss     = round(random.uniform(0.08, 0.35), 4)

        experiments.append({
            "exp_id"    : "EXP" + str(i + 1).zfill(3),
            "date"      : date.strftime("%Y-%m-%d"),
            "model"     : models[i % len(models)],
            "dataset"   : datasets[i % len(datasets)],
            "researcher": researchers[i % len(researchers)],
            "accuracy"  : accuracy,
            "f1_score"  : f1,
            "loss"      : loss,
            "epochs"    : random.choice([5, 8, 10, 15, 20, 25, 30])
        })

    return experiments


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def compute_trend(values):
    """
    Computes the linear trend direction and strength
    for a list of numeric values using simple linear regression.

    Returns:
        dict with slope, direction, and strength description
    """
    n = len(values)
    if n < 2:
        return {"slope": 0, "direction": "flat", "strength": "none"}

    x_mean = (n - 1) / 2
    y_mean = sum(values) / n

    numerator   = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator != 0 else 0

    direction = "improving" if slope > 0 else "declining" if slope < 0 else "flat"
    strength  = "strongly" if abs(slope) > 0.005 else "gradually" if abs(slope) > 0.001 else "slightly"

    return {
        "slope"    : round(slope, 6),
        "direction": direction,
        "strength" : strength
    }


def detect_anomalies(values, threshold=2.0):
    """
    Detects anomalous values that are more than threshold
    standard deviations away from the mean.

    Returns:
        list of (index, value) tuples for anomalous points
    """
    if len(values) < 3:
        return []

    mean   = statistics.mean(values)
    stdev  = statistics.stdev(values)

    if stdev == 0:
        return []

    anomalies = []
    for i, v in enumerate(values):
        z_score = abs(v - mean) / stdev
        if z_score > threshold:
            anomalies.append({"index": i, "value": v, "z_score": round(z_score, 2)})

    return anomalies


def compute_summary_stats(values):
    """
    Computes descriptive statistics for a list of values.
    Returns dict with mean, std, min, max, range, trend.
    """
    if not values:
        return {}

    return {
        "mean"   : round(statistics.mean(values), 4),
        "std"    : round(statistics.stdev(values) if len(values) > 1 else 0, 4),
        "min"    : round(min(values), 4),
        "max"    : round(max(values), 4),
        "range"  : round(max(values) - min(values), 4),
        "trend"  : compute_trend(values)
    }


# ============================================================
# TIME SERIES ANALYZER CLASS
# ============================================================

class TimeSeriesAnalyzer:
    """
    Analyzes ML experiment time series data and generates
    natural language insights using an LLM. Combines
    statistical analysis with LLM narration to produce
    human-readable reports about trends and patterns.
    """

    def __init__(self):
        self.client      = OllamaClient()
        self.daily_data  = generate_experiment_metrics(days=30)
        self.experiments = generate_experiment_runs(num_experiments=8)
        print("Time Series Analyzer ready.")
        print("Daily records    : " + str(len(self.daily_data)))
        print("Experiment runs  : " + str(len(self.experiments)))

    def _format_daily_summary(self):
        """Formats daily metrics as a compact text summary for LLM input."""
        accuracies = [r["accuracy"] for r in self.daily_data]
        losses     = [r["loss"]     for r in self.daily_data]
        f1_scores  = [r["f1_score"] for r in self.daily_data]

        acc_stats = compute_summary_stats(accuracies)
        loss_stats = compute_summary_stats(losses)

        lines = [
            "30-day ML experiment training metrics:",
            "Accuracy: start=" + str(accuracies[0]) +
            ", end=" + str(accuracies[-1]) +
            ", mean=" + str(acc_stats["mean"]) +
            ", trend=" + acc_stats["trend"]["strength"] + " " + acc_stats["trend"]["direction"],
            "Loss    : start=" + str(losses[0]) +
            ", end=" + str(losses[-1]) +
            ", mean=" + str(loss_stats["mean"]) +
            ", trend=" + loss_stats["trend"]["strength"] + " " + loss_stats["trend"]["direction"],
            "F1 Score: start=" + str(f1_scores[0]) +
            ", end=" + str(f1_scores[-1]),
            "Total improvement in accuracy: " +
            str(round(accuracies[-1] - accuracies[0], 4))
        ]
        return "\n".join(lines)

    def _format_experiment_summary(self):
        """Formats experiment run data as text for LLM input."""
        lines = ["Experiment runs summary:"]
        for exp in self.experiments:
            lines.append(
                exp["exp_id"] + ": " + exp["model"] +
                " by " + exp["researcher"] +
                " on " + exp["date"] +
                " | accuracy=" + str(exp["accuracy"]) +
                " f1=" + str(exp["f1_score"])
            )
        return "\n".join(lines)

    def analyze_training_trend(self):
        """
        Analyzes the 30-day training trend and asks the LLM
        to describe what the trend means for model performance.
        """
        summary = self._format_daily_summary()
        prompt  = (
            "You are a data analyst reviewing ML training progress. "
            "Analyze the following 30-day training metrics and describe "
            "the trend in 3 sentences. Mention whether training is "
            "converging, overfitting, or still improving.\n\n"
            + summary + "\n\nTrend Analysis:"
        )
        print("\n-- Training Trend Analysis --")
        print(summary)
        response = self.client.generate(MODEL, prompt)
        print("\nLLM Insight: " + response)
        return response

    def detect_and_explain_anomalies(self):
        """
        Detects statistical anomalies in accuracy values and
        asks the LLM to suggest possible causes.
        """
        accuracies = [r["accuracy"] for r in self.daily_data]
        anomalies  = detect_anomalies(accuracies, threshold=1.8)

        if not anomalies:
            print("\n-- Anomaly Detection --")
            print("No significant anomalies detected in accuracy values.")
            return []

        anomaly_str = ""
        for a in anomalies:
            day = self.daily_data[a["index"]]["date"]
            anomaly_str += (
                "Day " + day + ": accuracy=" + str(a["value"]) +
                " (z-score=" + str(a["z_score"]) + ")\n"
            )

        prompt = (
            "The following anomalous accuracy values were detected "
            "during ML model training. Suggest 2 possible causes "
            "for each anomaly in one sentence each.\n\n"
            "Anomalies detected:\n" + anomaly_str + "\nPossible causes:"
        )
        print("\n-- Anomaly Detection --")
        print("Anomalies found: " + str(len(anomalies)))
        print(anomaly_str)
        response = self.client.generate(MODEL, prompt)
        print("LLM Insight: " + response)
        return anomalies

    def compare_experiment_runs(self):
        """
        Compares multiple experiment runs and asks the LLM
        to identify the best performing configuration and why.
        """
        summary = self._format_experiment_summary()
        prompt  = (
            "Compare the following ML experiment runs and identify: "
            "1) The best performing model and why, "
            "2) The worst performing model and why, "
            "3) One recommendation for the next experiment.\n\n"
            + summary + "\n\nComparison:"
        )
        print("\n-- Experiment Run Comparison --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def generate_weekly_report(self):
        """
        Generates a weekly summary report covering the first 7 days
        of training metrics with LLM-written narrative sections.
        """
        week_data  = self.daily_data[:7]
        accuracies = [r["accuracy"] for r in week_data]
        losses     = [r["loss"]     for r in week_data]

        week_summary = (
            "Week 1 training summary (days 1-7):\n"
            "Accuracy range: " + str(min(accuracies)) + " to " + str(max(accuracies)) + "\n"
            "Loss range    : " + str(min(losses))     + " to " + str(max(losses))     + "\n"
            "Best accuracy : day " + str(accuracies.index(max(accuracies)) + 1) +
            " (" + str(max(accuracies)) + ")\n"
            "Worst accuracy: day " + str(accuracies.index(min(accuracies)) + 1) +
            " (" + str(min(accuracies)) + ")"
        )

        prompt = (
            "Write a short weekly ML training report based on these metrics. "
            "Include what went well and what needs attention. "
            "Keep it to 3 sentences.\n\n"
            + week_summary + "\n\nWeekly Report:"
        )
        print("\n-- Weekly Report Generation --")
        print(week_summary)
        response = self.client.generate(MODEL, prompt)
        print("\nGenerated Report: " + response)
        return response

    def forecast_next_week(self):
        """
        Uses the last 7 days of accuracy data to extrapolate
        a simple linear forecast for the next 7 days, then
        asks the LLM to comment on the projected performance.
        """
        recent     = [r["accuracy"] for r in self.daily_data[-7:]]
        trend      = compute_trend(recent)
        last_value = recent[-1]

        forecast = []
        for i in range(1, 8):
            predicted = round(min(0.99, last_value + trend["slope"] * i), 4)
            forecast.append(predicted)

        forecast_str = (
            "Last 7 days accuracy: " + ", ".join(str(v) for v in recent) + "\n"
            "Trend: " + trend["strength"] + " " + trend["direction"] + "\n"
            "Projected next 7 days: " + ", ".join(str(v) for v in forecast)
        )

        prompt = (
            "Based on the following ML training accuracy trend, "
            "comment on whether the model is likely to reach production "
            "quality (accuracy above 0.95) in the next 7 days. "
            "Give a confidence level and one recommendation.\n\n"
            + forecast_str + "\n\nForecast Commentary:"
        )
        print("\n-- 7-Day Accuracy Forecast --")
        print(forecast_str)
        response = self.client.generate(MODEL, prompt)
        print("\nLLM Commentary: " + response)
        return forecast


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("TIME SERIES ANALYZER DEMO")
    print("=" * 55)

    analyzer = TimeSeriesAnalyzer()

    print("\n-- Daily Metrics Sample (first 5 days) --")
    for record in analyzer.daily_data[:5]:
        print("  " + record["date"] +
              " | accuracy=" + str(record["accuracy"]) +
              " | loss=" + str(record["loss"]) +
              " | f1=" + str(record["f1_score"]))

    print("\n-- Statistical Summary --")
    accuracies = [r["accuracy"] for r in analyzer.daily_data]
    stats      = compute_summary_stats(accuracies)
    print("Accuracy stats: " + str(stats))

    analyzer.analyze_training_trend()
    analyzer.detect_and_explain_anomalies()
    analyzer.compare_experiment_runs()
    analyzer.generate_weekly_report()
    analyzer.forecast_next_week()

    print("\n-- Time Series Analyzer demo complete --")


if __name__ == "__main__":
    run_demo()
