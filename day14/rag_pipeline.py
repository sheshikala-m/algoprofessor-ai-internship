# ============================================================
# RAG PIPELINE
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Full Retrieval-Augmented Generation pipeline
# ============================================================

# This file ties everything together. I struggled to understand
# how retrieval connects to generation. The key insight is:
# retrieval finds relevant chunks, generation uses them as context.
# Without good retrieval, even a great LLM gives wrong answers.

from data_fetcher import DataFetcher
from document_processor import DocumentProcessor, filter_chunks
from embedding_generator import EmbeddingGenerator, find_most_similar, cosine_similarity
from vector_db_setup import get_vector_store

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ============================================================
# BM25 RETRIEVER (keyword-based)
# ============================================================

class BM25Retriever:
    """
    Keyword-based retrieval using BM25 algorithm.
    Good at finding exact keyword matches.
    Works without any ML models.
    """
    def __init__(self, documents):
        self.documents = documents
        if BM25_AVAILABLE:
            tokenized = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized)
            self._real = True
        else:
            self._real = False
            print("rank-bm25 not installed. Using TF-based fallback.")

    def _tf_score(self, query, doc):
        """Simple term frequency score as fallback."""
        query_words = set(query.lower().split())
        doc_words = doc.lower().split()
        matches = sum(1 for w in doc_words if w in query_words)
        return matches / (len(doc_words) + 1)

    def retrieve(self, query, top_k=5):
        """Returns list of (doc_text, score) tuples."""
        if self._real:
            scores = self.bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        else:
            scores = [(i, self._tf_score(query, doc)) for i, doc in enumerate(self.documents)]
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            results.append({"text": self.documents[idx], "score": float(score), "idx": idx})
        return results


# ============================================================
# SEMANTIC RETRIEVER (vector-based)
# ============================================================

class SemanticRetriever:
    """
    Dense vector retrieval using sentence embeddings.
    Good at finding semantically similar content even without
    exact keyword overlap.
    """
    def __init__(self, documents, metadatas=None):
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        self.gen = EmbeddingGenerator()
        print("Generating document embeddings...")
        self.embeddings = self.gen.generate(documents)

    def retrieve(self, query, top_k=5):
        q_emb = self.gen.generate_query(query)
        results = find_most_similar(q_emb, self.embeddings, self.documents, top_k=top_k)
        for i, r in enumerate(results):
            r["metadata"] = self.metadatas[i] if i < len(self.metadatas) else {}
        return results


# ============================================================
# HYBRID RETRIEVER (BM25 + Semantic + RRF fusion)
# ============================================================

def reciprocal_rank_fusion(bm25_results, semantic_results, k=60):
    """
    Combines BM25 and semantic results using Reciprocal Rank Fusion.
    RRF score = sum(1 / (rank + k)) across all result lists.
    I learned this technique from a paper on hybrid retrieval.
    """
    scores = {}
    text_map = {}

    for rank, result in enumerate(bm25_results):
        text = result["text"]
        scores[text] = scores.get(text, 0) + 1 / (rank + 1 + k)
        text_map[text] = result

    for rank, result in enumerate(semantic_results):
        text = result["text"]
        scores[text] = scores.get(text, 0) + 1 / (rank + 1 + k)
        text_map[text] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": text, "rrf_score": round(score, 6),
             "original": text_map[text]} for text, score in ranked]


class HybridRetriever:
    """
    Combines BM25 keyword search with semantic vector search.
    Uses RRF to merge results from both methods.
    Usually outperforms either method alone.
    """
    def __init__(self, documents, metadatas=None):
        self.bm25 = BM25Retriever(documents)
        self.semantic = SemanticRetriever(documents, metadatas)

    def retrieve(self, query, top_k=5):
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        semantic_results = self.semantic.retrieve(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion(bm25_results, semantic_results)
        return fused[:top_k]


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(query, retrieved_chunks, max_context_chars=1500):
    """
    Builds a prompt for the LLM using retrieved context.
    Truncates context if too long.
    """
    context_parts = []
    total_chars = 0
    for i, chunk in enumerate(retrieved_chunks):
        text = chunk.get("text", "")
        if total_chars + len(text) > max_context_chars:
            break
        context_parts.append(f"[{i+1}] {text}")
        total_chars += len(text)

    context = "\n\n".join(context_parts)
    prompt = (
        f"You are an ML experiment tracker assistant.\n"
        f"Use the following context to answer the question.\n"
        f"If the answer is not in the context, say 'Not found in records'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt


# ============================================================
# MOCK LLM GENERATOR
# ============================================================

def mock_generate(prompt):
    """
    Simulates LLM response by extracting key info from context.
    In production this would call Groq, OpenAI, or local model.
    """
    lines = prompt.split('\n')
    context_lines = [l for l in lines if l.startswith('[')]
    if not context_lines:
        return "Not found in records."
    # find query
    q_line = next((l for l in lines if l.startswith("Question:")), "")
    query_lower = q_line.lower()
    # return most relevant context line
    best_line = context_lines[0]
    best_score = 0
    for line in context_lines:
        words = set(query_lower.split())
        overlap = sum(1 for w in line.lower().split() if w in words)
        if overlap > best_score:
            best_score = overlap
            best_line = line
    return f"Based on records: {best_line.lstrip('[0123456789] ').strip()}"


# ============================================================
# FULL RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1. Fetch data
    2. Process and chunk documents
    3. Build retriever (hybrid by default)
    4. Answer queries with retrieved context
    """
    def __init__(self, retriever_type="hybrid"):
        self.retriever_type = retriever_type
        self.retriever = None
        self.chunks = []
        self.chunk_meta = []
        self._setup()

    def _setup(self):
        print("\n-- Setting up RAG Pipeline --")

        # Step 1: Fetch
        fetcher = DataFetcher(source="mock_db")
        texts, metadatas = fetcher.fetch()

        # Step 2: Process
        processor = DocumentProcessor()
        chunks, chunk_meta = processor.process_documents(texts, metadatas)
        chunks, chunk_meta = filter_chunks(chunks, chunk_meta, min_words=5)

        self.chunks = chunks
        self.chunk_meta = chunk_meta

        # Step 3: Build retriever
        if self.retriever_type == "hybrid":
            self.retriever = HybridRetriever(chunks, chunk_meta)
        elif self.retriever_type == "semantic":
            self.retriever = SemanticRetriever(chunks, chunk_meta)
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever(chunks)
        else:
            self.retriever = HybridRetriever(chunks, chunk_meta)

        print(f"RAG Pipeline ready. Chunks: {len(chunks)}, Retriever: {self.retriever_type}")

    def query(self, question, top_k=3, verbose=False):
        """Retrieves context and generates an answer."""
        results = self.retriever.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, results)
        answer = mock_generate(prompt)

        if verbose:
            print(f"\nQuery: {question}")
            print(f"Retrieved {len(results)} chunks")
            for i, r in enumerate(results):
                print(f"  [{i+1}] {r['text'][:80]}...")
            print(f"Answer: {answer}")
        return {"question": question, "answer": answer, "context": results}

    def batch_query(self, questions, top_k=3):
        return [self.query(q, top_k=top_k) for q in questions]


# ============================================================
# DEMO
# ============================================================

TEST_QUESTIONS = [
    "Which experiments did Ananya run?",
    "What was the best F1 score achieved?",
    "Which model performed best on NLP tasks?",
    "What datasets were used in CV experiments?",
    "How many epochs did EXP001 train for?",
]


def run_demo():
    print("=" * 55)
    print("RAG PIPELINE DEMO")
    print("=" * 55)

    pipeline = RAGPipeline(retriever_type="hybrid")

    print("\n-- Running test queries --")
    for q in TEST_QUESTIONS:
        result = pipeline.query(q, top_k=3, verbose=True)

    print("\n-- Comparing retriever types --")
    test_q = "Which experiments used transformer models?"
    for rtype in ["bm25", "semantic", "hybrid"]:
        p = RAGPipeline(retriever_type=rtype)
        r = p.query(test_q, top_k=2)
        print(f"\n{rtype.upper()}: {r['answer'][:100]}")

    print("\n-- RAG Pipeline demo complete --")


if __name__ == "__main__":
    run_demo()
