# ============================================================
# VECTOR DATABASE SETUP
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Set up ChromaDB as vector store for experiment docs
# ============================================================

# I spent a while figuring out why ChromaDB was not persisting
# data between runs. Turns out I had to pass the persist_directory
# argument correctly. Took me some time but works now.

import os
import json
import hashlib

# --- ChromaDB setup ---
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("chromadb not installed. Using in-memory mock store.")

# --- Embedding support ---
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

import numpy as np


# ============================================================
# MOCK VECTOR STORE (works without chromadb installed)
# ============================================================

class MockVectorStore:
    """
    Simulates a vector database using in-memory dicts.
    Same interface as the real ChromaDB wrapper below.
    I wrote this so I could test the logic before installing chromadb.
    """
    def __init__(self, collection_name="ml_experiments"):
        self.collection_name = collection_name
        self.documents = {}   # id -> text
        self.metadatas = {}   # id -> dict
        self.embeddings = {}  # id -> list[float]
        print(f"MockVectorStore: collection '{collection_name}' ready")

    def add_documents(self, docs, metadatas=None, ids=None):
        for i, doc in enumerate(docs):
            doc_id = ids[i] if ids else f"doc_{i}_{hashlib.md5(doc.encode()).hexdigest()[:6]}"
            self.documents[doc_id] = doc
            self.metadatas[doc_id] = metadatas[i] if metadatas else {}
            # simple fake embedding: character frequency vector of length 50
            vec = [doc.count(chr(97 + j % 26)) / max(len(doc), 1) for j in range(50)]
            self.embeddings[doc_id] = vec
        print(f"Added {len(docs)} documents. Total: {len(self.documents)}")

    def query(self, query_text, n_results=3):
        if not self.documents:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        # fake cosine similarity using dot product on fake embeddings
        q_vec = [query_text.count(chr(97 + j % 26)) / max(len(query_text), 1) for j in range(50)]
        scores = {}
        for doc_id, emb in self.embeddings.items():
            dot = sum(a * b for a, b in zip(q_vec, emb))
            norm_q = sum(x**2 for x in q_vec) ** 0.5
            norm_d = sum(x**2 for x in emb) ** 0.5
            scores[doc_id] = dot / (norm_q * norm_d + 1e-9)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        ids = [r[0] for r in ranked]
        return {
            "ids": [ids],
            "documents": [[self.documents[i] for i in ids]],
            "metadatas": [[self.metadatas[i] for i in ids]],
            "distances": [[1 - scores[i] for i in ids]]
        }

    def count(self):
        return len(self.documents)

    def delete_collection(self):
        self.documents.clear()
        self.metadatas.clear()
        self.embeddings.clear()
        print("Collection cleared.")


# ============================================================
# REAL CHROMADB WRAPPER
# ============================================================

class ChromaVectorStore:
    """
    Wraps ChromaDB with a simple add/query interface.
    Uses sentence-transformers for embeddings if available,
    otherwise falls back to ChromaDB's default embeddings.
    """
    def __init__(self, collection_name="ml_experiments", persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2") if EMBEDDINGS_AVAILABLE else None
        print(f"ChromaDB collection '{collection_name}' ready at '{persist_dir}'")

    def _embed(self, texts):
        if self.encoder:
            return self.encoder.encode(texts).tolist()
        return None

    def add_documents(self, docs, metadatas=None, ids=None):
        if not ids:
            ids = [f"doc_{hashlib.md5(d.encode()).hexdigest()[:8]}" for d in docs]
        embeddings = self._embed(docs)
        kwargs = dict(documents=docs, ids=ids)
        if metadatas:
            kwargs["metadatas"] = metadatas
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.add(**kwargs)
        print(f"Added {len(docs)} documents. Total: {self.collection.count()}")

    def query(self, query_text, n_results=3):
        q_emb = self._embed([query_text]) if self.encoder else None
        if q_emb:
            return self.collection.query(query_embeddings=q_emb, n_results=n_results)
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def count(self):
        return self.collection.count()

    def delete_collection(self):
        self.client.delete_collection(self.collection.name)
        print("Collection deleted.")


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_vector_store(collection_name="ml_experiments", persist_dir="./chroma_db"):
    """Returns real ChromaDB store if available, else mock store."""
    if CHROMA_AVAILABLE:
        return ChromaVectorStore(collection_name, persist_dir)
    return MockVectorStore(collection_name)


# ============================================================
# SAMPLE ML EXPERIMENT DOCUMENTS
# ============================================================

EXPERIMENT_DOCS = [
    "Experiment EXP001: Researcher Ananya trained BERT on NLP-Corpus-v2 dataset. "
    "Achieved accuracy 0.91, F1 0.89 after 10 epochs with batch size 32.",

    "Experiment EXP002: Researcher Vikram fine-tuned RoBERTa on SentimentData-v1. "
    "Final accuracy 0.94, precision 0.93, recall 0.95. Used learning rate 2e-5.",

    "Experiment EXP003: Researcher Priya trained ResNet50 on ImageNet-Subset. "
    "Top-1 accuracy 0.87, Top-5 accuracy 0.96. Training took 48 hours on 2 GPUs.",

    "Experiment EXP004: Researcher Rohan ran LSTM on TimeSeriesData-v3 for forecasting. "
    "MSE 0.023, MAE 0.14. Model converged after 25 epochs.",

    "Experiment EXP005: Researcher Ananya tested GPT-2 fine-tuning on CustomCorpus. "
    "BLEU score 0.72, ROUGE-L 0.68. Batch size 16, 5 epochs.",

    "Project NLP-Research uses datasets NLP-Corpus-v2 and SentimentData-v1. "
    "Focus: text classification and sentiment analysis. Lead: Ananya.",

    "Project CV-Research uses ImageNet-Subset and CIFAR-100. "
    "Focus: image recognition. Lead: Priya. Uses ResNet and EfficientNet.",

    "Dataset NLP-Corpus-v2: 500k documents, preprocessed, tokenized. "
    "Domain: news articles. Split: 80/10/10 train/val/test.",

    "Dataset SentimentData-v1: 100k reviews labeled positive/negative/neutral. "
    "Balanced classes. Collected from product reviews.",

    "Model BERT-base: 110M parameters, 12 transformer layers, pretrained on Wikipedia. "
    "Used in experiments EXP001 and EXP006 for classification tasks.",
]

EXPERIMENT_METADATA = [
    {"type": "experiment", "researcher": "Ananya", "model": "BERT"},
    {"type": "experiment", "researcher": "Vikram", "model": "RoBERTa"},
    {"type": "experiment", "researcher": "Priya", "model": "ResNet50"},
    {"type": "experiment", "researcher": "Rohan", "model": "LSTM"},
    {"type": "experiment", "researcher": "Ananya", "model": "GPT-2"},
    {"type": "project", "lead": "Ananya", "domain": "NLP"},
    {"type": "project", "lead": "Priya", "domain": "CV"},
    {"type": "dataset", "size": "500k", "domain": "NLP"},
    {"type": "dataset", "size": "100k", "domain": "Sentiment"},
    {"type": "model", "params": "110M", "architecture": "transformer"},
]


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("VECTOR DB SETUP DEMO")
    print("=" * 55)

    store = get_vector_store(collection_name="ml_experiments_demo")

    print("\n-- Loading experiment documents --")
    store.add_documents(
        docs=EXPERIMENT_DOCS,
        metadatas=EXPERIMENT_METADATA,
        ids=[f"exp_doc_{i}" for i in range(len(EXPERIMENT_DOCS))]
    )

    print(f"\nTotal documents in store: {store.count()}")

    queries = [
        "Which experiments used BERT?",
        "What accuracy did Ananya achieve?",
        "Show me NLP datasets",
        "Experiments with high F1 score",
    ]

    print("\n-- Running sample queries --")
    for q in queries:
        print(f"\nQuery: {q}")
        results = store.query(q, n_results=2)
        docs = results["documents"][0]
        for j, doc in enumerate(docs):
            print(f"  Result {j+1}: {doc[:90]}...")

    print("\n-- Vector DB setup complete --")


if __name__ == "__main__":
    run_demo()
