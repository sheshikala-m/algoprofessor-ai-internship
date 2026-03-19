# ============================================================
# TEST GROQ
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Test Groq LLM API as the generation step in RAG
# ============================================================

# Groq provides very fast LLM inference. I tested it as the
# generation component of the RAG pipeline. The API is similar
# to OpenAI so it was not too hard to integrate. The tricky part
# was handling the case where the API key is not set.

import os
import json


# ============================================================
# GROQ CLIENT SETUP
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# If groq package not installed, try direct HTTP
try:
    import urllib.request
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


# ============================================================
# GROQ GENERATOR
# ============================================================

class GroqGenerator:
    """
    Calls Groq API to generate answers using retrieved context.
    Falls back to mock generator if API key not set.
    Model: llama3-8b-8192 (fast and free on Groq)
    """
    def __init__(self, model="llama3-8b-8192", max_tokens=300):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = GROQ_API_KEY

        if not self.api_key:
            print("GROQ_API_KEY not set. Will use mock generator.")
            print("To use Groq: export GROQ_API_KEY=your_key_here")
            self._use_mock = True
        elif GROQ_AVAILABLE:
            self.client = Groq(api_key=self.api_key)
            self._use_mock = False
            print(f"Groq client ready. Model: {model}")
        else:
            print("groq package not installed. pip install groq")
            print("Falling back to mock generator.")
            self._use_mock = True

    def generate(self, prompt, system_prompt=None):
        """Generates a response from Groq or mock."""
        if self._use_mock:
            return self._mock_generate(prompt)
        try:
            return self._groq_generate(prompt, system_prompt)
        except Exception as e:
            print(f"Groq API error: {e}. Falling back to mock.")
            return self._mock_generate(prompt)

    def _groq_generate(self, prompt, system_prompt=None):
        """Real Groq API call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,   # low temperature for factual answers
        )
        return response.choices[0].message.content.strip()

    def _mock_generate(self, prompt):
        """Simulates LLM response when API not available."""
        lines = [l.strip() for l in prompt.split('\n') if l.strip()]
        context_lines = [l for l in lines if l.startswith('[') and ']' in l]
        q_line = next((l for l in lines if l.startswith("Question:")), "")
        query_words = set(q_line.lower().split())

        best_line = context_lines[0] if context_lines else "No context available."
        best_score = 0
        for line in context_lines:
            overlap = sum(1 for w in line.lower().split() if w in query_words)
            if overlap > best_score:
                best_score = overlap
                best_line = line

        text = best_line.split('] ', 1)[-1] if '] ' in best_line else best_line
        return f"[Mock] Based on records: {text[:200]}"


# ============================================================
# RAG WITH GROQ
# ============================================================

class RAGWithGroq:
    """
    RAG pipeline that uses Groq for the generation step.
    This replaces the mock_generate in rag_pipeline.py
    with a real LLM call.
    """
    def __init__(self):
        from rag_pipeline import HybridRetriever, build_prompt
        from data_fetcher import DataFetcher
        from document_processor import DocumentProcessor, filter_chunks

        self.build_prompt = build_prompt
        self.generator = GroqGenerator()

        print("\nSetting up RAG+Groq pipeline...")
        fetcher = DataFetcher(source="mock_db")
        texts, metadatas = fetcher.fetch()
        processor = DocumentProcessor()
        chunks, chunk_meta = processor.process_documents(texts, metadatas)
        chunks, chunk_meta = filter_chunks(chunks, chunk_meta, min_words=5)
        self.retriever = HybridRetriever(chunks, chunk_meta)
        self.chunks = chunks
        print(f"Pipeline ready. {len(chunks)} chunks indexed.")

    def query(self, question, top_k=3):
        """Full RAG query with Groq generation."""
        results = self.retriever.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, results)
        system = (
            "You are an ML experiment tracker assistant. "
            "Answer questions about experiments, models, datasets, and researchers. "
            "Be concise and factual. Base your answer only on the provided context."
        )
        answer = self.generator.generate(prompt, system_prompt=system)
        return {
            "question": question,
            "answer": answer,
            "num_sources": len(results),
            "used_groq": not self.generator._use_mock
        }


# ============================================================
# GROQ API TESTS
# ============================================================

def test_direct_api():
    """Tests Groq API with a simple standalone call."""
    if not GROQ_API_KEY:
        print("Skipping direct API test: GROQ_API_KEY not set.")
        return False

    print("\n-- Direct Groq API test --")
    gen = GroqGenerator()
    prompt = (
        "Context: Experiment EXP001 by Ananya used BERT model "
        "on NLP-Corpus-v2 dataset. Achieved accuracy 0.91 and F1 0.89.\n\n"
        "Question: What accuracy did Ananya achieve in EXP001?\n\nAnswer:"
    )
    response = gen.generate(prompt)
    print(f"Response: {response}")
    return True


def test_rag_with_groq():
    """Tests full RAG pipeline with Groq generation."""
    print("\n-- RAG + Groq pipeline test --")
    rag = RAGWithGroq()

    test_questions = [
        "Which experiments did Ananya run?",
        "What was the best accuracy in NLP experiments?",
        "Which researcher worked on time series forecasting?",
        "What model was used in EXP003?",
    ]

    for q in test_questions:
        result = rag.query(q, top_k=3)
        groq_tag = "[Groq]" if result["used_groq"] else "[Mock]"
        print(f"\n{groq_tag} Q: {result['question']}")
        print(f"  A: {result['answer'][:120]}")
        print(f"  Sources: {result['num_sources']} chunks")


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("GROQ LLM TEST DEMO")
    print("=" * 55)

    if GROQ_API_KEY:
        print(f"Groq API key found (length: {len(GROQ_API_KEY)})")
    else:
        print("No Groq API key. Tests will use mock generator.")
        print("Set GROQ_API_KEY environment variable to use real Groq.")

    test_direct_api()
    test_rag_with_groq()

    print("\n-- Groq test demo complete --")


if __name__ == "__main__":
    run_demo()
