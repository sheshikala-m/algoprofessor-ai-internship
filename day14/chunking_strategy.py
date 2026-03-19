# ============================================================
# CHUNKING STRATEGIES
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Different ways to split experiment documents into chunks
# ============================================================

# I did not understand why chunking matters at first.
# Then I realized if a document is too long, the embedding
# loses important details. If too short, there is no context.
# Getting the right chunk size is important for good retrieval.

import re
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# CHUNK DATA CLASS
# ============================================================

@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    strategy: str
    start_char: int = 0
    end_char: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# STRATEGY 1: FIXED SIZE CHUNKING
# ============================================================

def fixed_size_chunking(text, doc_id, chunk_size=200, overlap=50):
    """
    Splits text into chunks of fixed character size with overlap.
    Simple but can cut words/sentences in the middle.
    Good as a baseline.
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_fixed_{idx}",
                doc_id=doc_id,
                strategy="fixed_size",
                start_char=start,
                end_char=end
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks


# ============================================================
# STRATEGY 2: SENTENCE CHUNKING
# ============================================================

def sentence_chunking(text, doc_id, sentences_per_chunk=3, overlap=1):
    """
    Splits text into chunks of N sentences.
    Better than fixed size because chunks end at sentence boundaries.
    I prefer this for experiment notes that have full sentences.
    """
    # split on period, exclamation, or question mark followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    idx = 0
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i:i + sentences_per_chunk]
        chunk_text = " ".join(chunk_sents).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_sent_{idx}",
                doc_id=doc_id,
                strategy="sentence",
            ))
            idx += 1
        i += sentences_per_chunk - overlap
    return chunks


# ============================================================
# STRATEGY 3: PARAGRAPH CHUNKING
# ============================================================

def paragraph_chunking(text, doc_id):
    """
    Splits text on double newlines (paragraph boundaries).
    Best when documents are well-structured with paragraphs.
    Works well for README-style experiment descriptions.
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if para:
            chunks.append(Chunk(
                text=para,
                chunk_id=f"{doc_id}_para_{idx}",
                doc_id=doc_id,
                strategy="paragraph",
            ))
    return chunks


# ============================================================
# STRATEGY 4: RECURSIVE CHUNKING (LangChain style)
# ============================================================

def recursive_chunking(text, doc_id, chunk_size=300, overlap=50,
                        separators=None):
    """
    Tries to split on larger boundaries first (paragraphs, then
    sentences, then words, then characters). Falls back to next
    separator if chunk is still too large. This is what LangChain
    does internally. Took me a while to understand the recursion.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text, separators):
        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            # base case: split by character
            parts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            return [p for p in parts if p.strip()]

        parts = text.split(sep)
        result = []
        current = ""
        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += part + sep
            else:
                if current.strip():
                    if len(current) > chunk_size and next_seps:
                        result.extend(_split(current.strip(), next_seps))
                    else:
                        result.append(current.strip())
                current = part + sep
        if current.strip():
            result.append(current.strip())
        return result

    raw_chunks = _split(text, separators)
    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_rec_{idx}",
                doc_id=doc_id,
                strategy="recursive",
            ))
    return chunks


# ============================================================
# STRATEGY 5: SEMANTIC CHUNKING (keyword-based)
# ============================================================

SECTION_KEYWORDS = [
    "experiment", "result", "dataset", "model",
    "metric", "project", "researcher", "training",
    "evaluation", "conclusion"
]

def semantic_chunking(text, doc_id):
    """
    Splits text when a new topic/section keyword is detected.
    Groups sentences by topic. Not perfect but better than
    random splits for structured experiment reports.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    idx = 0

    for sent in sentences:
        sent_lower = sent.lower()
        is_new_section = any(kw in sent_lower for kw in SECTION_KEYWORDS)

        if is_new_section and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_sem_{idx}",
                    doc_id=doc_id,
                    strategy="semantic",
                ))
                idx += 1
            current_chunk = [sent]
        else:
            current_chunk.append(sent)

    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_sem_{idx}",
                doc_id=doc_id,
                strategy="semantic",
            ))
    return chunks


# ============================================================
# STRATEGY ROUTER
# ============================================================

def chunk_document(text, doc_id, strategy="sentence", **kwargs):
    """Routes to the correct chunking strategy."""
    strategies = {
        "fixed_size": fixed_size_chunking,
        "sentence": sentence_chunking,
        "paragraph": paragraph_chunking,
        "recursive": recursive_chunking,
        "semantic": semantic_chunking,
    }
    if strategy not in strategies:
        print(f"Unknown strategy '{strategy}'. Using 'sentence'.")
        strategy = "sentence"
    return strategies[strategy](text, doc_id, **kwargs)


# ============================================================
# DEMO
# ============================================================

SAMPLE_DOC = """
Experiment EXP001: Researcher Ananya trained BERT on the NLP-Corpus-v2 dataset.
The experiment ran for 10 epochs with batch size 32 and learning rate 2e-5.
Final accuracy was 0.91 and F1 score was 0.89.

The model used dropout of 0.1 to prevent overfitting.
Ananya noted that increasing batch size beyond 32 did not help.
Early stopping was used with patience of 3 epochs.

Dataset NLP-Corpus-v2 contains 500k documents from news articles.
The split was 80% training, 10% validation, and 10% test.
Text was tokenized using BERT tokenizer with max length 512.

Results were compared against a baseline TF-IDF + SVM model.
BERT achieved 12% improvement in F1 over the baseline.
Next steps: try RoBERTa and experiment with different learning rates.
"""


def run_demo():
    print("=" * 55)
    print("CHUNKING STRATEGIES DEMO")
    print("=" * 55)

    strategies = ["fixed_size", "sentence", "paragraph", "recursive", "semantic"]

    for strat in strategies:
        chunks = chunk_document(SAMPLE_DOC.strip(), doc_id="EXP001", strategy=strat)
        print(f"\nStrategy: {strat} -> {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] ({len(c.text)} chars) {c.text[:80]}...")

    print("\n-- Best strategy for experiment docs: sentence or recursive --")
    print("-- Fixed size is too rigid; paragraph needs well-structured docs --")


if __name__ == "__main__":
    run_demo()
