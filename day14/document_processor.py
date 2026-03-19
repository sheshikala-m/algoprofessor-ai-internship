# ============================================================
# DOCUMENT PROCESSOR
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Clean, normalize and prepare documents before embedding
# ============================================================

# I realized document quality affects retrieval quality a lot.
# If the text has lots of noise like extra spaces or weird chars,
# the embeddings become less useful. Cleaning is important.

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from data_fetcher import DataFetcher, ExperimentRecord
from chunking_strategy import chunk_document, Chunk


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def remove_extra_whitespace(text):
    """Collapses multiple spaces/newlines into single ones."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_unicode(text):
    """Converts unicode to closest ASCII equivalent."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def remove_special_chars(text, keep_punctuation=True):
    """Removes non-alphanumeric chars, optionally keeps punctuation."""
    if keep_punctuation:
        # keep letters, digits, spaces, and common punctuation
        text = re.sub(r'[^\w\s.,;:()\-\'/]', ' ', text)
    else:
        text = re.sub(r'[^\w\s]', ' ', text)
    return remove_extra_whitespace(text)

def lowercase_text(text):
    return text.lower()

def truncate_text(text, max_chars=2000):
    """Truncates text to max_chars, trying to end at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated + "..."

def fix_metric_formatting(text):
    """
    Normalizes metric values in experiment text.
    e.g., 'accuracy=0.910000' -> 'accuracy=0.91'
    I added this because some records had overly precise floats.
    """
    def round_float(match):
        try:
            val = float(match.group())
            return f"{val:.4f}".rstrip('0').rstrip('.')
        except Exception:
            return match.group()
    return re.sub(r'\d+\.\d{5,}', round_float, text)


# ============================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================

class DocumentProcessor:
    """
    Cleans, normalizes, and chunks raw text documents.
    Applies a configurable pipeline of cleaning steps.
    """

    def __init__(self, config=None):
        default_config = {
            "lowercase": False,        # lowercase before embedding? keeps proper nouns intact
            "remove_special": True,    # remove special chars
            "normalize_unicode": True, # normalize unicode
            "fix_metrics": True,       # fix float formatting
            "max_chars": 2000,         # max doc length before chunking
            "chunk_strategy": "sentence",
            "chunk_kwargs": {"sentences_per_chunk": 3, "overlap": 1},
        }
        self.config = config or default_config
        self._processed = 0
        self._failed = 0

    def clean(self, text):
        """Applies all enabled cleaning steps to a single text."""
        if not text or not text.strip():
            return ""
        if self.config.get("normalize_unicode"):
            text = normalize_unicode(text)
        if self.config.get("fix_metrics"):
            text = fix_metric_formatting(text)
        if self.config.get("remove_special"):
            text = remove_special_chars(text, keep_punctuation=True)
        if self.config.get("lowercase"):
            text = lowercase_text(text)
        text = remove_extra_whitespace(text)
        text = truncate_text(text, self.config.get("max_chars", 2000))
        return text

    def process_documents(self, texts, metadatas=None):
        """
        Cleans and chunks a list of documents.
        Returns list of (chunk_text, chunk_metadata) tuples.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        all_chunks = []
        all_metadata = []

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            try:
                cleaned = self.clean(text)
                if not cleaned:
                    self._failed += 1
                    continue

                doc_id = meta.get("exp_id", f"doc_{i}")
                chunks = chunk_document(
                    cleaned,
                    doc_id=doc_id,
                    strategy=self.config.get("chunk_strategy", "sentence"),
                    **self.config.get("chunk_kwargs", {})
                )

                for chunk in chunks:
                    chunk_meta = dict(meta)
                    chunk_meta["chunk_id"] = chunk.chunk_id
                    chunk_meta["strategy"] = chunk.strategy
                    chunk_meta["chunk_index"] = len(all_chunks)
                    all_chunks.append(chunk.text)
                    all_metadata.append(chunk_meta)

                self._processed += 1

            except Exception as e:
                print(f"Error processing doc {i}: {e}")
                self._failed += 1

        print(f"Processed {self._processed} docs -> {len(all_chunks)} chunks. "
              f"Failed: {self._failed}")
        return all_chunks, all_metadata

    def get_stats(self):
        return {"processed": self._processed, "failed": self._failed}


# ============================================================
# QUALITY FILTER
# ============================================================

def filter_chunks(chunks, metadatas, min_words=5, max_words=200):
    """
    Removes chunks that are too short or too long.
    Short chunks usually have no useful content.
    Very long chunks dilute the embedding.
    """
    filtered_chunks = []
    filtered_meta = []
    removed = 0
    for chunk, meta in zip(chunks, metadatas):
        word_count = len(chunk.split())
        if min_words <= word_count <= max_words:
            filtered_chunks.append(chunk)
            filtered_meta.append(meta)
        else:
            removed += 1
    print(f"Quality filter: kept {len(filtered_chunks)}, removed {removed} chunks")
    return filtered_chunks, filtered_meta


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("DOCUMENT PROCESSOR DEMO")
    print("=" * 55)

    # fetch raw data
    fetcher = DataFetcher(source="mock_db")
    texts, metadatas = fetcher.fetch()

    print(f"\nRaw documents: {len(texts)}")
    print(f"Sample raw text: {texts[0][:100]}...")

    # test cleaning on a noisy text
    noisy = "  Experiment   EXP001!!   accuracy=0.910000   researcher: Ananya\n\n\n  "
    processor = DocumentProcessor()
    cleaned = processor.clean(noisy)
    print(f"\nCleaning test:")
    print(f"  Before: '{noisy.strip()}'")
    print(f"  After:  '{cleaned}'")

    # process all documents
    print("\n-- Processing all documents --")
    chunks, chunk_meta = processor.process_documents(texts, metadatas)

    # apply quality filter
    chunks, chunk_meta = filter_chunks(chunks, chunk_meta, min_words=5)

    print(f"\nFinal chunks: {len(chunks)}")
    print("\nSample chunks:")
    for i in range(min(3, len(chunks))):
        print(f"\n[{i+1}] {chunks[i]}")
        print(f"    Meta: {chunk_meta[i]}")

    print("\n-- Document Processor demo complete --")


if __name__ == "__main__":
    run_demo()
