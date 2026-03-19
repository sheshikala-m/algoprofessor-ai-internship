# ============================================================
# DATA FETCHER
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Load and prepare ML experiment data for RAG pipeline
# ============================================================

# I was confused initially about why we need a data fetcher
# separately from the document processor. Turns out fetcher
# handles WHERE data comes from (files, DB, API) and processor
# handles HOW to clean and split it. Makes sense once I saw it.

import os
import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ExperimentRecord:
    """Represents a single ML experiment record."""
    exp_id: str
    researcher: str
    project: str
    dataset: str
    model: str
    epochs: int
    batch_size: int
    learning_rate: float
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_text(self):
        """Converts record to natural language for embedding."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"Experiment {self.exp_id} by {self.researcher} on project {self.project}. "
            f"Used {self.model} model on {self.dataset} dataset. "
            f"Trained for {self.epochs} epochs with batch size {self.batch_size} "
            f"and learning rate {self.learning_rate}. "
            f"Results: {metrics_str}. "
            f"Notes: {self.notes}"
        )

    def to_metadata(self):
        return {
            "exp_id": self.exp_id,
            "researcher": self.researcher,
            "project": self.project,
            "dataset": self.dataset,
            "model": self.model,
            "accuracy": self.metrics.get("accuracy", 0.0),
            "f1": self.metrics.get("f1", 0.0),
        }


# ============================================================
# MOCK DATABASE (simulates SQL rows)
# ============================================================

MOCK_DB_RECORDS = [
    ExperimentRecord(
        exp_id="EXP001", researcher="Ananya", project="NLP-Research",
        dataset="NLP-Corpus-v2", model="BERT", epochs=10, batch_size=32,
        learning_rate=2e-5,
        metrics={"accuracy": 0.91, "f1": 0.89, "precision": 0.90, "recall": 0.88},
        notes="Best result so far on NLP-Corpus. Tried dropout 0.1."
    ),
    ExperimentRecord(
        exp_id="EXP002", researcher="Vikram", project="NLP-Research",
        dataset="SentimentData-v1", model="RoBERTa", epochs=8, batch_size=16,
        learning_rate=1e-5,
        metrics={"accuracy": 0.94, "f1": 0.93, "precision": 0.93, "recall": 0.95},
        notes="RoBERTa outperformed BERT on sentiment. Will try on more datasets."
    ),
    ExperimentRecord(
        exp_id="EXP003", researcher="Priya", project="CV-Research",
        dataset="ImageNet-Subset", model="ResNet50", epochs=30, batch_size=64,
        learning_rate=0.001,
        metrics={"accuracy": 0.87, "top5_accuracy": 0.96, "loss": 0.34},
        notes="Used data augmentation. Helped with overfitting."
    ),
    ExperimentRecord(
        exp_id="EXP004", researcher="Rohan", project="TimeSeries-Research",
        dataset="TimeSeriesData-v3", model="LSTM", epochs=25, batch_size=128,
        learning_rate=0.01,
        metrics={"mse": 0.023, "mae": 0.14, "r2": 0.91},
        notes="Tried different sequence lengths. 30-step window worked best."
    ),
    ExperimentRecord(
        exp_id="EXP005", researcher="Ananya", project="NLP-Research",
        dataset="CustomCorpus", model="GPT-2", epochs=5, batch_size=16,
        learning_rate=5e-5,
        metrics={"bleu": 0.72, "rouge_l": 0.68, "perplexity": 18.4},
        notes="Fine-tuning GPT-2 is tricky. Overfits quickly, needed early stopping."
    ),
    ExperimentRecord(
        exp_id="EXP006", researcher="Vikram", project="NLP-Research",
        dataset="NLP-Corpus-v2", model="DistilBERT", epochs=12, batch_size=32,
        learning_rate=3e-5,
        metrics={"accuracy": 0.89, "f1": 0.88, "inference_time_ms": 12.3},
        notes="DistilBERT is 40% faster than BERT with only 3% accuracy drop."
    ),
    ExperimentRecord(
        exp_id="EXP007", researcher="Priya", project="CV-Research",
        dataset="CIFAR-100", model="EfficientNet-B0", epochs=50, batch_size=32,
        learning_rate=0.0005,
        metrics={"accuracy": 0.82, "top5_accuracy": 0.94, "params_M": 5.3},
        notes="EfficientNet-B0 much lighter than ResNet50 with decent accuracy."
    ),
    ExperimentRecord(
        exp_id="EXP008", researcher="Rohan", project="TimeSeries-Research",
        dataset="TimeSeriesData-v3", model="Transformer", epochs=20, batch_size=64,
        learning_rate=0.0001,
        metrics={"mse": 0.018, "mae": 0.11, "r2": 0.94},
        notes="Transformer beats LSTM on this dataset. Attention helps capture long deps."
    ),
]

MOCK_PROJECT_DOCS = [
    {"project": "NLP-Research", "lead": "Ananya", "domain": "NLP",
     "datasets": ["NLP-Corpus-v2", "SentimentData-v1", "CustomCorpus"],
     "description": "Research on text classification, sentiment analysis, and language generation."},
    {"project": "CV-Research", "lead": "Priya", "domain": "Computer Vision",
     "datasets": ["ImageNet-Subset", "CIFAR-100"],
     "description": "Image recognition and classification using CNNs."},
    {"project": "TimeSeries-Research", "lead": "Rohan", "domain": "Time Series",
     "datasets": ["TimeSeriesData-v3"],
     "description": "Forecasting and anomaly detection on time series data."},
]


# ============================================================
# DATA FETCHER CLASS
# ============================================================

class DataFetcher:
    """
    Fetches ML experiment data from various sources.
    Supports mock DB, JSON files, and CSV files.
    Returns list of (text, metadata) tuples ready for embedding.
    """

    def __init__(self, source="mock_db"):
        self.source = source
        print(f"DataFetcher initialized with source: {source}")

    def fetch_from_mock_db(self):
        """Loads hardcoded experiment records."""
        records = MOCK_DB_RECORDS
        texts = [r.to_text() for r in records]
        metadatas = [r.to_metadata() for r in records]

        # also add project-level documents
        for proj in MOCK_PROJECT_DOCS:
            text = (
                f"Project {proj['project']} led by {proj['lead']}. "
                f"Domain: {proj['domain']}. "
                f"Uses datasets: {', '.join(proj['datasets'])}. "
                f"{proj['description']}"
            )
            texts.append(text)
            metadatas.append({"type": "project", "project": proj["project"],
                               "lead": proj["lead"], "domain": proj["domain"]})

        print(f"Fetched {len(texts)} documents from mock DB.")
        return texts, metadatas

    def fetch_from_json(self, filepath):
        """Loads experiment records from a JSON file."""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return [], []
        with open(filepath) as f:
            data = json.load(f)
        texts, metadatas = [], []
        for item in data:
            rec = ExperimentRecord(**item)
            texts.append(rec.to_text())
            metadatas.append(rec.to_metadata())
        print(f"Fetched {len(texts)} documents from {filepath}.")
        return texts, metadatas

    def fetch_from_csv(self, filepath):
        """Loads experiment records from a CSV file."""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return [], []
        texts, metadatas = [], []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    f"Experiment {row.get('exp_id', 'N/A')} by {row.get('researcher', 'N/A')}. "
                    f"Model: {row.get('model', 'N/A')}. "
                    f"Dataset: {row.get('dataset', 'N/A')}. "
                    f"Accuracy: {row.get('accuracy', 'N/A')}."
                )
                texts.append(text)
                metadatas.append({k: row[k] for k in ["exp_id", "researcher", "model", "dataset"]
                                   if k in row})
        print(f"Fetched {len(texts)} documents from {filepath}.")
        return texts, metadatas

    def fetch(self, filepath=None):
        """Main fetch method - routes to correct source."""
        if self.source == "mock_db":
            return self.fetch_from_mock_db()
        elif self.source == "json" and filepath:
            return self.fetch_from_json(filepath)
        elif self.source == "csv" and filepath:
            return self.fetch_from_csv(filepath)
        else:
            print(f"Unknown source '{self.source}', falling back to mock DB.")
            return self.fetch_from_mock_db()

    def export_to_json(self, filepath="experiments.json"):
        """Exports mock DB records to JSON for later use."""
        data = []
        for rec in MOCK_DB_RECORDS:
            data.append({
                "exp_id": rec.exp_id,
                "researcher": rec.researcher,
                "project": rec.project,
                "dataset": rec.dataset,
                "model": rec.model,
                "epochs": rec.epochs,
                "batch_size": rec.batch_size,
                "learning_rate": rec.learning_rate,
                "metrics": rec.metrics,
                "notes": rec.notes,
            })
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} records to {filepath}")


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("DATA FETCHER DEMO")
    print("=" * 55)

    fetcher = DataFetcher(source="mock_db")
    texts, metadatas = fetcher.fetch()

    print(f"\nTotal documents fetched: {len(texts)}")
    print("\nSample documents:")
    for i, (text, meta) in enumerate(zip(texts[:3], metadatas[:3])):
        print(f"\n[{i+1}] {text[:120]}...")
        print(f"    Metadata: {meta}")

    print("\n-- Exporting to JSON --")
    fetcher.export_to_json("experiments_export.json")

    print("\n-- Fetching from exported JSON --")
    fetcher2 = DataFetcher(source="json")
    texts2, _ = fetcher2.fetch("experiments_export.json")
    print(f"Loaded {len(texts2)} records from JSON.")

    # cleanup
    if os.path.exists("experiments_export.json"):
        os.remove("experiments_export.json")

    print("\n-- Data Fetcher demo complete --")


if __name__ == "__main__":
    run_demo()
