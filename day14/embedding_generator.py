# ============================================================
# EMBEDDING GENERATOR
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Generate vector embeddings for experiment documents
# ============================================================

# Embeddings were the most confusing concept for me initially.
# The idea that a sentence becomes a list of 384 numbers and
# two similar sentences have similar number lists is fascinating.
# I finally understood it when I visualized cosine similarity.

import math
import hashlib
from typing import List, Tuple


# ============================================================
# SENTENCE TRANSFORMERS (real embeddings)
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("sentence-transformers not installed. Using mock embeddings.")


# ============================================================
# MOCK EMBEDDER (works without sentence-transformers)
# ============================================================

class MockEmbedder:
    """
    Generates deterministic fake embeddings using character statistics.
    Dimension is fixed at 64 (much smaller than real 384-dim embeddings).
    Two documents with similar words will have similar embeddings.
    """
    def __init__(self, dim=64):
        self.dim = dim
        self.model_name = "mock-embedder"
        print(f"MockEmbedder ready (dim={dim}). Install sentence-transformers for real embeddings.")

    def _embed_one(self, text):
        """Creates a fake embedding vector from text."""
        text = text.lower()
        vec = []
        # use character n-gram frequencies as features
        for i in range(self.dim):
            char_idx = i % 26
            char = chr(ord('a') + char_idx)
            count = text.count(char)
            # add positional variation
            val = (count + i * 0.01) / (len(text) + 1)
            vec.append(val)
        # normalize to unit vector
        norm = math.sqrt(sum(x**2 for x in vec)) + 1e-9
        return [x / norm for x in vec]

    def encode(self, texts, batch_size=32, show_progress=False):
        return [self._embed_one(t) for t in texts]

    def get_sentence_embedding_dimension(self):
        return self.dim


# ============================================================
# REAL EMBEDDER WRAPPER
# ============================================================

class SentenceEmbedder:
    """
    Wraps sentence-transformers SentenceTransformer.
    Falls back to MockEmbedder if not installed.
    I used all-MiniLM-L6-v2 because it is small and fast.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if ST_AVAILABLE:
            print(f"Loading model: {model_name} ...")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.dim}")
        else:
            self.model = MockEmbedder()
            self.model_name = self.model.model_name
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=32, show_progress=False):
        """Returns list of embedding vectors."""
        if ST_AVAILABLE:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )
            return embeddings.tolist()
        else:
            return self.model.encode(texts)

    def encode_query(self, query):
        """Encodes a single query string."""
        return self.encode([query])[0]

    def get_dim(self):
        return self.dim


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec_a, vec_b):
    """Computes cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x**2 for x in vec_a)) + 1e-9
    norm_b = math.sqrt(sum(x**2 for x in vec_b)) + 1e-9
    return dot / (norm_a * norm_b)

def find_most_similar(query_emb, doc_embeddings, doc_texts, top_k=3):
    """Finds top-k most similar documents to a query embedding."""
    scores = []
    for i, emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_emb, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in scores[:top_k]:
        results.append({
            "text": doc_texts[idx],
            "score": round(score, 4),
            "rank": len(results) + 1
        })
    return results


# ============================================================
# EMBEDDING CACHE (avoids re-embedding same text)
# ============================================================

class EmbeddingCache:
    """
    Simple in-memory cache. Stores text hash -> embedding.
    Useful when running experiments multiple times with same docs.
    """
    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def _key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text):
        key = self._key(text)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, text, embedding):
        self._cache[self._key(text)] = embedding

    def stats(self):
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": round(hit_rate, 3)}


# ============================================================
# CACHED EMBEDDING GENERATOR
# ============================================================

class EmbeddingGenerator:
    """
    Combines SentenceEmbedder with EmbeddingCache.
    Skips re-embedding texts that were already embedded.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceEmbedder(model_name)
        self.cache = EmbeddingCache()

    def generate(self, texts, batch_size=32):
        """Generates embeddings, using cache when available."""
        results = [None] * len(texts)
        to_embed = []
        indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append(text)
                indices.append(i)

        if to_embed:
            new_embeddings = self.embedder.encode(to_embed, batch_size=batch_size)
            for idx, emb in zip(indices, new_embeddings):
                self.cache.set(texts[idx], emb)
                results[idx] = emb

        print(f"Generated {len(to_embed)} new embeddings. Cache stats: {self.cache.stats()}")
        return results

    def generate_query(self, query):
        return self.embedder.encode_query(query)


# ============================================================
# DEMO
# ============================================================

SAMPLE_TEXTS = [
    "Experiment EXP001: Ananya trained BERT on NLP-Corpus-v2. Accuracy 0.91, F1 0.89.",
    "Experiment EXP002: Vikram fine-tuned RoBERTa on SentimentData. Accuracy 0.94.",
    "Experiment EXP003: Priya trained ResNet50 on ImageNet. Top-1 accuracy 0.87.",
    "Experiment EXP004: Rohan ran LSTM on TimeSeriesData. MSE 0.023.",
    "Project NLP-Research focuses on text classification and sentiment analysis.",
    "Dataset NLP-Corpus-v2 has 500k documents from news articles.",
]


def run_demo():
    print("=" * 55)
    print("EMBEDDING GENERATOR DEMO")
    print("=" * 55)

    gen = EmbeddingGenerator()

    print("\n-- Generating embeddings for sample docs --")
    embeddings = gen.generate(SAMPLE_TEXTS)
    print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0])}")

    print("\n-- Second pass (should use cache) --")
    embeddings2 = gen.generate(SAMPLE_TEXTS)

    print("\n-- Similarity search --")
    queries = [
        "Which researcher worked on BERT?",
        "What was the accuracy of NLP experiments?",
        "Show me computer vision experiments",
    ]
    for q in queries:
        q_emb = gen.generate_query(q)
        results = find_most_similar(q_emb, embeddings, SAMPLE_TEXTS, top_k=2)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  [{r['rank']}] score={r['score']} | {r['text'][:70]}...")

    print("\n-- Cosine similarity between related docs --")
    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    sim_04 = cosine_similarity(embeddings[0], embeddings[4])
    print(f"EXP001 vs EXP002 (both NLP):    {sim_01:.4f}")
    print(f"EXP001 vs EXP003 (different):   {sim_02:.4f}")
    print(f"EXP001 vs Project desc (related): {sim_04:.4f}")

    print("\n-- Embedding Generator demo complete --")


if __name__ == "__main__":
    run_demo()
