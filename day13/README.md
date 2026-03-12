Day 13: Vector Databases

Author: Sheshikala
Date: March 13, 2026

Overview

For Day 13, I explored vector databases and how they enable semantic search by storing text as embeddings. Instead of relying on keyword matching, vector search compares the meaning of text by measuring similarity between numerical vectors.

To keep the workflow consistent with earlier days, I used machine learning experiment descriptions created previously and indexed them for semantic retrieval.

How to Run

Install the required packages:

pip install -r requirements.txt

Run the scripts in the following order:

python embedding_gen.py
python vector_db_setup.py
python hybrid_search.py
python interface_with_pinecone.py

All scripts run locally and do not require Docker.

Description of Each Script

embedding_gen.py

This script demonstrates how embeddings are generated and used for similarity search.

Key components include:

Loading the all-MiniLM-L6-v2 embedding model from the sentence-transformers library

Generating both single and batch embeddings

Computing cosine similarity between experiment descriptions

Implementing simple semantic search using NumPy operations

Splitting long documents into overlapping text chunks for better retrieval performance

vector_db_setup.py

This script demonstrates two different vector database approaches.

Chroma is used for a simple setup where text, embeddings, and metadata are stored together.
FAISS is used for manual embedding indexing and provides faster similarity search for larger datasets.

The script also includes a comparison of both systems.

hybrid_search.py

This script compares three retrieval approaches:

BM25: Keyword-based search that performs well for exact terms

Semantic search: Embedding-based retrieval that captures meaning and paraphrases

Hybrid search (RRF): Combines both approaches using Reciprocal Rank Fusion

Several query types are tested to observe differences in retrieval quality.

interface_with_pinecone.py

This script demonstrates how a cloud-based vector database works using Pinecone.

It runs in demonstration mode without an API key but prints the operations that would normally occur in a real environment.
To use a live Pinecone database, create a free account and add your API key.

Key Concepts Learned

Embeddings

Embeddings convert text into dense numerical vectors that represent semantic meaning.
Texts with similar meaning produce vectors that are closer in vector space.

The model all-MiniLM-L6-v2 generates embeddings with 384 dimensions.
It is important to use the same embedding model during both indexing and querying to maintain consistency.

Chroma vs FAISS

Chroma is easier to use and supports metadata filtering. It can persist data using SQLite and is suitable for prototyping and retrieval-augmented generation systems.

FAISS requires manual embedding generation and metadata handling but provides highly optimized similarity search and scales well to very large datasets.

BM25 vs Semantic vs Hybrid Retrieval

BM25 performs well when the query contains exact keywords.
Semantic search performs better when queries are paraphrased or expressed differently.
Hybrid search combines both approaches and often provides the most reliable results.

Text Chunking

Long documents are split into smaller chunks before generating embeddings.
Chunking helps maintain search accuracy and prevents large documents from exceeding embedding limits.

A typical configuration uses approximately 200 words per chunk with about 50 words of overlap.
Overlap ensures that important context is not lost at chunk boundaries.

Challenges Encountered

Understanding how embeddings represent meaning in high-dimensional vector space required additional study.

FAISS also requires manual vector normalization when using cosine similarity, whereas Chroma handles this automatically.

The Reciprocal Rank Fusion formula required some experimentation before its purpose became clear.

Setting up Pinecone in serverless mode also required additional reading to understand its workflow.

Connection to Other Days

The experiment descriptions used for indexing were generated earlier and stored in MongoDB during Day 12.

Day 14 will extend this work by building a Retrieval-Augmented Generation pipeline using Chroma as a knowledge base.

Day 15 will further extend the concept with Graph-RAG using Neo4j to incorporate relationships between documents.
