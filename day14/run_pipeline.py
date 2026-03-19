# ============================================================
# RUN PIPELINE
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Master script to run all Day 14 components in order
# ============================================================

# This script runs all parts of the Day 14 pipeline in sequence.
# I use this to test everything end-to-end quickly without
# running each file separately. Helps catch integration issues.

import sys
import traceback
from datetime import datetime


def run_step(step_name, func):
    """Runs a step and catches any errors without stopping the whole pipeline."""
    print(f"\n{'='*55}")
    print(f"STEP: {step_name}")
    print(f"{'='*55}")
    try:
        func()
        print(f"[OK] {step_name} completed successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] {step_name} failed: {e}")
        traceback.print_exc()
        return False


def step_vector_db():
    from vector_db_setup import run_demo
    run_demo()


def step_data_fetcher():
    from data_fetcher import run_demo
    run_demo()


def step_chunking():
    from chunking_strategy import run_demo
    run_demo()


def step_document_processor():
    from document_processor import run_demo
    run_demo()


def step_embedding_generator():
    from embedding_generator import run_demo
    run_demo()


def step_rag_pipeline():
    from rag_pipeline import run_demo
    run_demo()


def step_qa_app():
    from qa_app import run_demo
    run_demo()


def step_groq_test():
    """
    Runs Groq LLM test if API key is configured.
    Falls back gracefully if key not set.
    """
    from test_groq import run_demo
    run_demo()


# ============================================================
# MAIN RUNNER
# ============================================================

ALL_STEPS = [
    ("Vector DB Setup",        step_vector_db),
    ("Data Fetcher",           step_data_fetcher),
    ("Chunking Strategies",    step_chunking),
    ("Document Processor",     step_document_processor),
    ("Embedding Generator",    step_embedding_generator),
    ("RAG Pipeline",           step_rag_pipeline),
    ("QA App",                 step_qa_app),
    ("Groq LLM Test",          step_groq_test),
]


def main(steps=None):
    """
    Runs all steps or a specific subset.
    steps: list of step names to run. If None, runs all.
    """
    print("=" * 55)
    print("DAY 14: ML EXPERIMENT TRACKER RAG PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    to_run = ALL_STEPS
    if steps:
        to_run = [(name, fn) for name, fn in ALL_STEPS if name in steps]

    results = {}
    for step_name, func in to_run:
        success = run_step(step_name, func)
        results[step_name] = "OK" if success else "FAILED"

    # summary
    print(f"\n{'='*55}")
    print("PIPELINE RUN SUMMARY")
    print(f"{'='*55}")
    passed = sum(1 for v in results.values() if v == "OK")
    failed = sum(1 for v in results.values() if v == "FAILED")
    for name, status in results.items():
        print(f"  [{status}] {name}")
    print(f"\nTotal: {passed} passed, {failed} failed")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    # you can pass step names as command line args to run only specific steps
    # e.g. python run_pipeline.py "Data Fetcher" "RAG Pipeline"
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        print(f"Running selected steps: {selected}")
        main(steps=selected)
    else:
        main()
