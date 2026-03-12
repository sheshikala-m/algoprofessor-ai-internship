"""
Day 13: Vector Database Setup - Chroma + FAISS
Author: Sheshikala
Date: 2026-03-13
"""

import time
import numpy as np


EXPERIMENT_DOCS = [
    {
        "id": "exp_001",
        "text": "BERT fine-tuned for sentiment analysis on movie reviews achieving 90.2 F1 score using AdamW optimizer",
        "metadata": {"model": "BERT", "task": "sentiment", "f1": 0.9020},
    },
    {
        "id": "exp_002",
        "text": "RoBERTa model trained for opinion classification outperforming BERT with 93.6 F1 score",
        "metadata": {"model": "RoBERTa", "task": "sentiment", "f1": 0.9360},
    },
    {
        "id": "exp_003",
        "text": "DistilBERT lightweight transformer for fast sentiment inference",
        "metadata": {"model": "DistilBERT", "task": "sentiment", "f1": 0.8985},
    },
    {
        "id": "exp_004",
        "text": "ResNet50 transfer learning for chest X-ray disease classification",
        "metadata": {"model": "ResNet50", "task": "medical_imaging", "f1": 0.8530},
    },
    {
        "id": "exp_005",
        "text": "DenseNet121 architecture for lung nodule detection",
        "metadata": {"model": "DenseNet121", "task": "medical_imaging", "f1": 0.8780},
    },
    {
        "id": "exp_006",
        "text": "XGBoost gradient boosting for credit card fraud detection",
        "metadata": {"model": "XGBoost", "task": "fraud_detection", "f1": 0.9279},
    },
    {
        "id": "exp_007",
        "text": "LightGBM gradient boosting for sales forecasting",
        "metadata": {"model": "LightGBM", "task": "forecasting", "f1": 0.9015},
    },
    {
        "id": "exp_008",
        "text": "GPT2 model fine-tuned for generating product descriptions",
        "metadata": {"model": "GPT2", "task": "text_generation", "f1": 0.7840},
    },
]


# CHROMA DATABASE
def demo_chromadb():

    print("\nCHROMA DATABASE")

    try:
        import chromadb
    except ImportError:
        print("chromadb not installed. Run: pip install chromadb")
        return

    client = chromadb.Client()

    collection = client.create_collection(
        name="ml_experiments",
        metadata={"hnsw:space": "cosine"},
    )

    print("Chroma collection created")

    collection.add(
        ids=[doc["id"] for doc in EXPERIMENT_DOCS],
        documents=[doc["text"] for doc in EXPERIMENT_DOCS],
        metadatas=[doc["metadata"] for doc in EXPERIMENT_DOCS],
    )

    print("Indexed documents:", collection.count())

    print("\nSemantic search example")

    results = collection.query(
        query_texts=["transformer model for text classification"],
        n_results=3,
    )

    for i, (doc_id, doc, dist) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0]),
        1,
    ):
        print(i, doc_id, "distance:", round(dist, 4))
        print(doc[:80])

    print("\nFiltered search example")

    results = collection.query(
        query_texts=["best performing NLP model"],
        n_results=3,
        where={"task": "sentiment"},
    )

    for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
        print(doc_id, doc[:70])

    doc = collection.get(ids=["exp_001"])

    print("\nGet document by id")
    print(doc["ids"][0], doc["documents"][0][:60])

    print("Chroma demo complete")

    return collection


# FAISS DATABASE
def demo_faiss():

    print("\nFAISS DATABASE")

    try:
        import faiss
    except ImportError:
        print("faiss not installed. Run: pip install faiss-cpu")
        return

    np.random.seed(42)

    dim = 384
    num_docs = len(EXPERIMENT_DOCS)

    embeddings = np.random.randn(num_docs, dim).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    print("FAISS index created")
    print("Total vectors:", index.ntotal)

    id_map = {i: doc["id"] for i, doc in enumerate(EXPERIMENT_DOCS)}

    query_vec = np.random.randn(1, dim).astype("float32")

    faiss.normalize_L2(query_vec)

    print("\nFAISS similarity search")

    t0 = time.time()

    scores, idxs = index.search(query_vec, 3)

    elapsed = (time.time() - t0) * 1000

    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), 1):
        doc_id = id_map[idx]
        print(rank, doc_id, "score:", round(score, 4))

    print("Search time:", round(elapsed, 3), "ms")

    print("\nIVF Index example")

    nlist = 2

    quantizer = faiss.IndexFlatIP(dim)

    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    ivf_index.train(embeddings)

    ivf_index.add(embeddings)

    ivf_index.nprobe = 1

    scores, idxs = ivf_index.search(query_vec, 3)

    print("IVF top result:", id_map[idxs[0][0]])

    print("Total vectors:", ivf_index.ntotal)

    print("FAISS demo complete")

    return index


# COMPARISON
def compare_chroma_faiss():

    print("\nChroma vs FAISS comparison")

    comparison = [
        ("Storage", "Persistent", "In memory"),
        ("Embeddings", "Built in", "Manual"),
        ("Metadata", "Supported", "Not supported"),
        ("Scale", "Millions", "Billions"),
        ("Ease", "Easy", "More setup"),
        ("Speed", "Good", "Very fast"),
    ]

    print("Property | Chroma | FAISS")

    for prop, chroma, faiss_val in comparison:
        print(prop, "|", chroma, "|", faiss_val)


# MAIN
if __name__ == "__main__":

    print("Day 13: Vector Database Setup")
    print("=" * 40)

    demo_chromadb()

    demo_faiss()

    compare_chroma_faiss()

    print("\nvector_db_setup.py completed")
