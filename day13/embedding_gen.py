"""
Day 13: Embedding Generation
Author: Sheshikala
Date: 2026-03-13
"""

import time
import numpy as np
from typing import List


# LOAD MODEL
def load_model():
    try:
        from sentence_transformers import SentenceTransformer

        print("Loading sentence-transformers model...")
        print("(first run downloads about 90MB)")

        model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Model loaded successfully")
        print("Embedding dimension:", model.get_sentence_embedding_dimension())

        return model

    except ImportError:
        print("sentence-transformers not installed")
        print("Run: pip install sentence-transformers")
        exit(1)


# SINGLE EMBEDDING
def demo_single_embedding(model):

    print("\n--- Single Embedding ---")

    text = "BERT fine-tuned for sentiment analysis achieving 90 percent F1 score"

    t0 = time.time()
    emb = model.encode(text)
    t1 = time.time()

    print("Text:", text)
    print("Shape:", emb.shape)
    print("Dtype:", emb.dtype)
    print("Sample:", emb[:5].round(4))
    print("Time:", (t1 - t0) * 1000, "ms")
    print("Norm:", np.linalg.norm(emb))


# BATCH EMBEDDINGS
def demo_batch_embeddings(model):

    print("\n--- Batch Embeddings ---")

    texts = [
        "BERT sentiment classification with transformer architecture",
        "RoBERTa improved pretraining for NLP tasks",
        "XGBoost gradient boosting for tabular fraud detection",
        "ResNet50 convolutional network for medical image classification",
        "LightGBM fast gradient boosting for time series forecasting",
        "DistilBERT smaller faster BERT for production deployment",
        "DenseNet deep convolutional network for lung nodule detection",
        "GPT-2 autoregressive language model for text generation",
    ]

    t0 = time.time()

    embeddings = model.encode(texts, batch_size=4, show_progress_bar=False)

    t1 = time.time()

    print("Encoded", len(texts), "texts")
    print("Shape:", embeddings.shape)
    print("Time:", (t1 - t0) * 1000, "ms total")

    return texts, embeddings


# COSINE SIMILARITY
def demo_similarity(model, texts, embeddings):

    print("\n--- Cosine Similarity ---")

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    bert_emb = embeddings[0]
    roberta_emb = embeddings[1]
    xgboost_emb = embeddings[2]

    sim_bert_roberta = cosine_sim(bert_emb, roberta_emb)
    sim_bert_xgboost = cosine_sim(bert_emb, xgboost_emb)

    print("BERT vs RoBERTa similarity:", sim_bert_roberta)
    print("BERT vs XGBoost similarity:", sim_bert_xgboost)

    from sentence_transformers import util

    sim_matrix = util.cos_sim(embeddings, embeddings).numpy()

    print("\nSimilarity matrix (first 4x4):")

    labels = ["BERT", "RoBERTa", "XGBoost", "ResNet50"]

    print("           ", end="")
    for l in labels:
        print(l, "      ", end="")
    print()

    for i, l1 in enumerate(labels):
        print(l1, "   ", end="")
        for j in range(4):
            print(round(sim_matrix[i][j], 4), "   ", end="")
        print()

    print("\nMost similar pairs:")

    pairs = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append((sim_matrix[i][j], texts[i][:30], texts[j][:30]))

    pairs.sort(reverse=True)

    for score, t1, t2 in pairs[:3]:
        print(round(score, 4), "|", t1, "<->", t2)


# SEMANTIC SEARCH
def demo_semantic_search(model, texts, embeddings):

    print("\n--- Semantic Search ---")

    def search(query, top_k=3):

        query_emb = model.encode(query)

        similarities = []

        for i, emb in enumerate(embeddings):
            score = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append((score, texts[i]))

        return sorted(similarities, reverse=True)[:top_k]

    queries = [
        "NLP model for classifying opinions",
        "fast tree based method for tabular data",
        "deep learning for medical diagnosis",
    ]

    for q in queries:

        print("\nQuery:", q)

        for rank, (score, text) in enumerate(search(q), 1):
            print("#", rank, "|", round(score, 4), "|", text)


# CHUNKING
def demo_chunking():

    print("\n--- Text Chunking ---")

    long_document = """
    Experiment Report: Sentiment Analysis Project
    
    We trained multiple transformer models on the IMDB movie review dataset containing 50000 samples.
    The dataset was split into 40000 training samples and 10000 test samples.
    
    Model 1: BERT-base-uncased achieved 90.2 percent F1 score.
    Model 2: RoBERTa achieved 93.6 percent F1 score.
    Model 3: DistilBERT achieved 89.85 percent F1 score with faster inference.
    
    Conclusion: RoBERTa best accuracy, DistilBERT best speed.
    """

    def chunk_text(text: str, chunk_size: int = 60, overlap: int = 10) -> List[str]:

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = min(start + chunk_size, len(words))

            chunk = " ".join(words[start:end])

            chunks.append(chunk)

            if end == len(words):
                break

            start = end - overlap

        return chunks

    chunks = chunk_text(long_document)

    print("Chunks created:", len(chunks))

    for i, chunk in enumerate(chunks, 1):
        print("\nChunk", i)
        print(chunk[:120])


# MAIN
if __name__ == "__main__":

    print("Day 13: Embedding Generation")
    print("=" * 40)

    model = load_model()

    demo_single_embedding(model)

    texts, embeddings = demo_batch_embeddings(model)

    demo_similarity(model, texts, embeddings)

    demo_semantic_search(model, texts, embeddings)

    demo_chunking()

    print("\nembedding_gen.py completed")
