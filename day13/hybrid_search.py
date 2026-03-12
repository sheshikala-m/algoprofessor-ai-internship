"""
Day 13: Hybrid Search - BM25 + Semantic + RRF Fusion
Author: Sheshikala
Date: 2026-03-13
"""

import numpy as np
from typing import List, Dict, Tuple


CORPUS = [
    {"id": "exp_001", "text": "BERT fine-tuned for sentiment analysis on movie reviews achieving 90.2 F1 using AdamW optimizer"},
    {"id": "exp_002", "text": "RoBERTa model trained for opinion classification outperforming BERT with 93.6 F1 score"},
    {"id": "exp_003", "text": "DistilBERT lightweight transformer smaller than BERT for fast sentiment inference in production"},
    {"id": "exp_004", "text": "ResNet50 transfer learning for chest X-ray disease classification in medical imaging"},
    {"id": "exp_005", "text": "DenseNet121 architecture for lung nodule detection and medical image segmentation"},
    {"id": "exp_006", "text": "XGBoost gradient boosting for credit card fraud detection on imbalanced dataset"},
    {"id": "exp_007", "text": "LightGBM gradient boosting for sales forecasting using transaction data"},
    {"id": "exp_008", "text": "GPT-2 language model fine-tuned for generating product descriptions"},
    {"id": "exp_009", "text": "LSTM recurrent network for time series prediction of stock prices"},
    {"id": "exp_010", "text": "Random Forest ensemble method for tabular data classification"},
]


# BM25 INDEX
def build_bm25_index(corpus: List[Dict]):

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank-bm25 not installed. Run: pip install rank-bm25")
        exit(1)

    tokenized = [doc["text"].lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized)

    return bm25


def bm25_search(bm25, query: str, corpus: List[Dict], top_k: int = 5):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [(corpus[i]["id"], corpus[i]["text"], score) for i, score in ranked]


# SEMANTIC INDEX
def build_semantic_index(corpus: List[Dict]):

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in corpus]

    embeddings = model.encode(texts, show_progress_bar=False)

    return model, embeddings


def semantic_search(model, embeddings, query: str, corpus: List[Dict], top_k: int = 5):

    query_emb = model.encode(query)

    similarities = []

    for i, emb in enumerate(embeddings):

        score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))

        similarities.append((i, float(score)))

    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    return [(corpus[i]["id"], corpus[i]["text"], score) for i, score in ranked]


# RRF FUSION
def reciprocal_rank_fusion(bm25_results: List[Tuple], semantic_results: List[Tuple], k: int = 60):

    scores = {}

    for rank, (doc_id, text, _) in enumerate(bm25_results, 1):

        if doc_id not in scores:
            scores[doc_id] = {"text": text, "rrf": 0.0}

        scores[doc_id]["rrf"] += 1.0 / (k + rank)

    for rank, (doc_id, text, _) in enumerate(semantic_results, 1):

        if doc_id not in scores:
            scores[doc_id] = {"text": text, "rrf": 0.0}

        scores[doc_id]["rrf"] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1]["rrf"], reverse=True)

    return [(doc_id, info["text"], info["rrf"]) for doc_id, info in ranked]


# COMPARE METHODS
def compare_methods(bm25, model, embeddings):

    print("\nComparing BM25 vs Semantic vs Hybrid")

    test_queries = [
        "transformer model for text classification",
        "XGBoost fraud",
        "deep learning lung detection",
        "fast lightweight model for production",
    ]

    for query in test_queries:

        print("\nQuery:", query)

        bm25_res = bm25_search(bm25, query, CORPUS, 5)

        semantic_res = semantic_search(model, embeddings, query, CORPUS, 5)

        hybrid_res = reciprocal_rank_fusion(bm25_res, semantic_res)[:5]

        print("Top Hybrid Results:")

        for rank, (doc_id, text, score) in enumerate(hybrid_res[:3], 1):

            print(rank, doc_id, "score:", round(score, 6), "|", text[:60])


# MAIN
if __name__ == "__main__":

    print("Day 13: Hybrid Search")
    print("=" * 40)

    print("Building BM25 index...")
    bm25 = build_bm25_index(CORPUS)
    print("BM25 index built")

    print("Building semantic index...")
    model, embeddings = build_semantic_index(CORPUS)
    print("Semantic index built")

    query = "neural network for image classification"

    print("\nBM25 results:")
    for rank, (doc_id, text, score) in enumerate(bm25_search(bm25, query, CORPUS, 3), 1):

        print(rank, doc_id, "score:", round(score, 4), "|", text[:60])

    print("\nSemantic results:")
    for rank, (doc_id, text, score) in enumerate(semantic_search(model, embeddings, query, CORPUS, 3), 1):

        print(rank, doc_id, "score:", round(score, 4), "|", text[:60])

    bm25_res = bm25_search(bm25, query, CORPUS, 5)

    semantic_res = semantic_search(model, embeddings, query, CORPUS, 5)

    hybrid_res = reciprocal_rank_fusion(bm25_res, semantic_res)

    print("\nHybrid RRF results:")
    for rank, (doc_id, text, score) in enumerate(hybrid_res[:3], 1):

        print(rank, doc_id, "score:", round(score, 6), "|", text[:60])

    compare_methods(bm25, model, embeddings)

    print("\nhybrid_search.py completed")
