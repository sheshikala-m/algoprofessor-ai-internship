"""
main.py — DataSense AI Entry Point
===================================
Runs the complete multi-agent analytics pipeline.

Usage:
    python main.py
    python main.py --data ecommerce_sales.csv
"""

import os
import time
import argparse
from datetime import datetime

# ── Agent Imports ─────────────────────────────────────────────
from scraper_agent import ScraperAgent
from data_analyst import DataAnalystAgent
from statistics_engine import StatisticsEngineAgent
from ml_orchestrator import MLOrchestratorAgent
from forecasting_agent import ForecastingAgent
from insight_reporter import InsightReporterAgent

OUTPUT_DIR = "outputs"


# ─────────────────────────────────────────────────────────────
# TERMINAL UI
# ─────────────────────────────────────────────────────────────
def print_banner():
    print("\n" + "=" * 70)
    print(" DataSense AI - Multi-Agent Analytics Platform ")
    print("=" * 70)


def print_step(step, title):
    print("\n" + "-" * 65)
    print(f" STEP {step}/6 | {title}")
    print("-" * 65)


def print_success(msg):
    print(f" [SUCCESS] {msg}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(data_path):

    start_time = time.time()
    output_files = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_banner()

    print(f"\nPipeline started: {datetime.now()}")
    print(f"Dataset: {data_path}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")

    # ── STEP 1: SCRAPER ──────────────────────────────────────
    print_step(1, "ScraperAgent - Market Data")

    scraper = ScraperAgent(output_dir=OUTPUT_DIR)
    scraped_df = scraper.generate_mock_scraped_data()

    print_success(f"Generated {len(scraped_df)} mock market records")

    # ── STEP 2: DATA ANALYST ─────────────────────────────────
    print_step(2, "DataAnalyst - EDA")

    analyst = DataAnalystAgent(output_dir=OUTPUT_DIR)

    analyst.load_data(data_path)
    analyst.clean_data()
    analyst.run_eda()

    heatmap_path = analyst.plot_correlation_heatmap()
    dist_path = analyst.plot_distributions()
    profile_path = analyst.export_profile_json()

    output_files.extend([
        heatmap_path,
        dist_path,
        profile_path
    ])

    print_success("EDA complete")

    analyst_msg = analyst.get_agent_message()

    # ── STEP 3: STATISTICS ENGINE ────────────────────────────
    print_step(3, "StatisticsEngine - Statistical Analysis")

    stats_engine = StatisticsEngineAgent(output_dir=OUTPUT_DIR)

    stats_engine.receive_data(analyst_msg)

    stats_engine.normality_tests()
    stats_engine.detect_outliers()
    stats_engine.correlation_significance()

    boxplot_path = stats_engine.plot_boxplots()
    stats_path = stats_engine.export_stats_json()

    output_files.extend([
        boxplot_path,
        stats_path
    ])

    print_success("Statistics complete")

    stats_msg = stats_engine.get_agent_message()

    # ── STEP 4: ML ORCHESTRATOR ──────────────────────────────
    print_step(4, "MLOrchestrator - ML Pipeline")

    ml_agent = MLOrchestratorAgent(output_dir=OUTPUT_DIR)

    ml_agent.receive_data(stats_msg)

    ml_agent.preprocess_features(target_col="high_value")
    ml_agent.run_pca(n_components=5)
    ml_agent.run_kmeans(n_clusters=4)
    ml_agent.run_random_forest()
    ml_agent.run_svm()

    ml_path = ml_agent.export_ml_json()

    output_files.append(ml_path)

    print_success("ML pipeline complete")

    ml_msg = ml_agent.get_agent_message()

    # ── STEP 5: FORECASTING ──────────────────────────────────
    print_step(5, "ForecastingAgent - Forecasting")

    forecaster = ForecastingAgent(output_dir=OUTPUT_DIR)

    forecaster.receive_data(
        analyst.df,
        date_col="date",
        value_col="sales_amount"
    )

    forecaster.aggregate_daily()
    forecaster.compute_rolling_stats(window=7)

    forecaster.forecast_polynomial(
        forecast_days=30,
        degree=3
    )

    forecaster.forecast_sma(window=7)

    forecast_chart = forecaster.plot_forecast()
    forecast_json = forecaster.export_forecast_json()

    output_files.extend([
        forecast_chart,
        forecast_json
    ])

    print_success("Forecast complete")

    # ── STEP 6: REPORTING ────────────────────────────────────
    print_step(6, "InsightReporter - Final Reports")

    reporter = InsightReporterAgent(output_dir=OUTPUT_DIR)

    full_context = {
        "data_analyst": analyst.profile,
        "statistics": stats_engine.stats_report,
        "ml": ml_agent.ml_results,
        "forecast": forecaster.forecast_results,
    }

    reporter.receive_context(full_context)

    md_path = reporter.generate_markdown_report()
    json_path = reporter.generate_json_summary()

    output_files.extend([
        md_path,
        json_path
    ])

    print_success("Reports generated")

    # ── FINAL SUMMARY ────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE ")
    print("=" * 70)

    for file in output_files:
        print(f"Generated: {file}")

    print(f"\nRuntime: {elapsed} seconds")
    print("\nDataSense AI completed successfully.\n")

    return output_files


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="ecommerce_sales.csv",
        help="CSV dataset path"
    )

    args = parser.parse_args()

    run_pipeline(data_path=args.data)