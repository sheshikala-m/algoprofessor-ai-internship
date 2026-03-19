# Day 7: RAG Pipeline for ML Experiment Tracker

**Author:** Sheshikala  
**Topic:** Build a full Retrieval-Augmented Generation pipeline over ML experiment data

---

## What I Built

A complete RAG pipeline that lets you ask questions about ML experiments, researchers, datasets, and models in natural language. The pipeline retrieves the most relevant experiment records and uses an LLM to generate factual answers.

---

## Folder Structure

```
day/
  vector_db_setup.py      - Sets up ChromaDB as the vector store
  data_fetcher.py         - Loads ML experiment data (mock DB, JSON, CSV)
  chunking_strategy.py    - 5 strategies: fixed, sentence, paragraph, recursive, semantic
  document_processor.py   - Cleans, normalizes, and chunks documents
  embedding_generator.py  - Generates sentence embeddings with caching
  rag_pipeline.py         - Full hybrid RAG pipeline (BM25 + semantic + RRF)
  qa_app.py               - Q&A app with session tracking and query enhancement
  run_pipeline.py         - Master script to run all components in sequence
  test_groq.py            - Tests Groq LLM as the generation step
  requirements.txt        - Python dependencies
  README.md               - This file
```

---

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run all components:
```
python run_pipeline.py
```

Run individual files:
```
python vector_db_setup.py
python data_fetcher.py
python chunking_strategy.py
python document_processor.py
python embedding_generator.py
python rag_pipeline.py
python qa_app.py
python test_groq.py
```

To use Groq LLM (optional):
```
set GROQ_API_KEY=your_groq_api_key_here
python test_groq.py
```

All files work without Groq API key using mock generators.

---

## Key Concepts Learned

| Concept | What I Learned |
|---|---|
| Chunking | Sentence and recursive chunking work best for experiment docs |
| Embeddings | Similar sentences have similar vectors (high cosine similarity) |
| BM25 | Keyword search, good for exact matches like experiment IDs |
| Semantic Search | Vector search, finds related content even without exact keywords |
| Hybrid + RRF | Combining both methods almost always beats either alone |
| RAG | Retrieval gives LLM accurate context, reduces hallucination |
| Groq | Fast LLM inference API, simple to integrate |

