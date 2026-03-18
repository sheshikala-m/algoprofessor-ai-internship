# algoprofessor-ai-internship
AI R&D Internship — AlgoProfessor AI R&D Solutions (Feb–May 2026)
Intern: Sheshikala Mamidisetti | Track: Data Science & ML Agentic AI
IIT Indore Drishti CPS AI & Data Science Certification (Pursuing)

## ML Algorithm Coverage
Linear/Logistic Regression | Decision Tree | Random Forest | SVM
K-Means | PCA | LDA | SVD | XGBoost | LightGBM | PyTorch DL
ChromaDB | FAISS | BM25 | RAG | Graph RAG | Neo4j

## Progress Tracker
- [x] Day 01: Python Fundamentals — EDA Pipeline, OOP, NumPy, Pandas, Matplotlib
- [x] Day 02: Machine Learning — Random Forest, SVM, KMeans, Model Comparison
- [x] Day 03: Feature Engineering — XGBoost, LightGBM, PCA, LDA, SVD, PyTorch NN
- [x] Day 04: Deep Neural Networks — CNN, Transfer Learning, Training Pipeline
- [x] Day 05: NLP — HuggingFace Transformers, Ollama, CoT, ReAct, DSPy
- [x] Day 06: LLM APIs — OpenAI, Claude, Function Calling, Pydantic, Memory
- [x] Day 07: RAG — FAISS, Embeddings, Retrieval Augmented Generation
- [x] Day 08: Agent — ReAct Agent, LangChain, Tool Use
- [x] Day 09: Multi-Agent — EDA Systems, CrewAI
- [x] Day 10: Milestone Project — DataOracle Capstone
- [x] Day 11: Databases — PostgreSQL, SQLAlchemy, pgvector, JSONB, Partitioning
- [x] Day 12: MongoDB + Redis — Document Store, Aggregation, Caching, Celery
- [x] Day 7: Vector DBs — ChromaDB, FAISS, Pinecone, Hybrid Search BM25+Semantic
- [x] Day 14: RAG Pipeline — Chunking, Embeddings, Hybrid Retrieval, RAGAS Eval
- [x] Day 15: Graph RAG — Neo4j Knowledge Graph, HyDE, Reranking, Streaming QA
- [ ] Phase 2: LLM Engineering, Fine-tuning on Tabular/Time Series Data
- [ ] Phase 3: Agentic AI + Grand Capstone: DataSense AI

## Weekly Progress Update — Week 1 (Feb 22 – Mar 1)
### Completed Work
- Day 01: Iris Data Engineering Pipeline
- OOP-based Analysis Engine with Visualization
- Automated Heatmap, Boxplot, and Pairplot generation
- Day 02: Machine Learning workflow initiated using Breast Cancer dataset

### Milestones Progress
- Phase 1 Foundations — In Progress
- ML workflow initialization completed

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Python for Data Science
- Exploratory Data Analysis (EDA)
- Data Visualization Techniques
- Introduction to Machine Learning
- Model Training & Evaluation Concepts

### Learning Outcomes
- Built automated data analysis pipelines
- Applied Object-Oriented Programming for analytics
- Initiated ML classification workflow
- Practiced reproducible project structuring using GitHub

---

## Weekly Progress Update — Week 2 (Mar 2 – Mar 8)
### Completed Work
**Day 02 — Machine Learning Models (Heart Disease & Breast Cancer Dataset)**
- Linear Regression and Logistic Regression on Breast Cancer dataset
- Decision Tree, Random Forest — Accuracy 0.80, ROC-AUC 0.91
- SVM Classification — Accuracy 0.82, ROC-AUC 0.883
- KMeans Clustering — Silhouette Score 0.167, Optimal K=2
- Hyperparameter Tuning on Decision Tree
- Model Comparison — SVM best overall accuracy

**Day 03 — Feature Engineering & Advanced ML (Wine Quality Dataset)**
- XGBoost Accuracy 0.825, ROC-AUC 0.881 vs LightGBM Accuracy 0.790
- Feature Engineering — 4 new features created, +1.56% accuracy improvement
- Dimensionality Reduction — PCA, LDA, SVD — Best: LDA Accuracy 0.7125
- Production-grade Scikit-learn Pipeline — Best: RF Accuracy 0.803, ROC-AUC 0.902
- PyTorch Neural Network from Scratch — Accuracy 0.759
- Auto Report Generation

### Milestones Progress
- Phase 1 Foundations — In Progress
- M1: Web Intelligence Synthesiser — In Progress
- All supervised and unsupervised ML models completed
- Advanced deep learning and pipelines initiated

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Supervised Learning — Classification and Regression
- Unsupervised Learning — Clustering
- Model Evaluation and Comparison
- Ensemble Methods — XGBoost and LightGBM
- Feature Engineering and Selection
- Dimensionality Reduction — PCA, LDA, SVD
- Deep Learning Foundations with PyTorch
- Production ML Pipeline Design

### Learning Outcomes
- Implemented 7 ML models with full evaluation metrics
- Applied ensemble methods XGBoost and LightGBM
- Built engineered features improving accuracy by +1.56%
- Reduced dimensions using PCA, LDA and SVD techniques
- Built production-grade Scikit-learn pipelines
- Implemented Neural Network from scratch using PyTorch
- Practiced professional GitHub commit workflow

---

## Weekly Progress Update — Week 3 (Mar 8 – Mar 14)
### Completed Work
**Day 11 — PostgreSQL Databases (ML Experiment Tracker)**
- Schema design for researchers, projects, datasets, experiments, metrics
- Complex SQL queries — JOINs, CTEs, Window Functions, aggregations
- JSONB storage for flexible ML hyperparameters with GIN indexing
- Table partitioning by date range and hash partitioning
- Advanced indexing strategies — B-tree, GIN, partial indexes
- SQLAlchemy async integration with pgvector extension

**Day 12 — MongoDB + Redis (ML Experiment Logs)**
- MongoDB CRUD operations on experiment collections
- Aggregation pipelines — $group, $facet, $lookup for F1 score analysis
- Redis cache-aside pattern — 200x speedup on repeated queries
- TTL-based cache expiry for experiment metadata
- Celery task queue for async experiment processing
- Cache benchmark comparing MongoDB vs Redis latency

**Day 13 — Vector Databases (ML Knowledge Base)**
- ChromaDB setup with cosine similarity and persistent storage
- FAISS IndexFlatL2 for exact search and IndexIVFFlat for approximate search
- Pinecone interface with demo mode fallback
- Hybrid search combining BM25 keyword search with semantic vector search
- Reciprocal Rank Fusion (RRF) merging BM25 and semantic rankings
- Embedding generation with sentence-transformers all-MiniLM-L6-v2

**Day 14 — RAG Pipeline (ML Experiment Tracker Knowledge Base)**
- 5 chunking strategies — fixed size, sentence, paragraph, recursive, semantic
- Document ingestion pipeline — load, clean, chunk, embed, store
- Basic RAG loop — index, retrieve, generate with retrieval recall evaluation
- Hybrid RAG pipeline with BM25 + semantic search and RRF fusion
- RAGAS-style evaluation — faithfulness, answer relevancy, context precision, recall
- Groq LLM integration with mock fallback for generation step

**Day 15 — Graph RAG + Advanced RAG (ML Knowledge Graph)**
- ML knowledge graph with 5 node types — Researcher, Project, Dataset, Experiment, Metric
- Multi-hop graph queries — Researcher to Experiment to Metric traversal
- 5 advanced RAG techniques — HyDE, query expansion, cross-encoder reranking,
  multi-query retrieval, contextual compression
- Streaming QA app with token-by-token output simulation
- Multi-turn conversation with context-aware retrieval
- Recall@1 and Recall@3 evaluation on 7 test cases

### Milestones Progress
- M2: Enterprise Knowledge Navigator — Completed
- PostgreSQL, MongoDB, Redis, Vector DBs, RAG, Graph RAG all implemented
- Full Week 3 knowledge pipeline from raw SQL to Graph RAG completed

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Database Systems — Relational and NoSQL
- Information Retrieval and Search
- Natural Language Processing — Embeddings and Semantic Search
- Knowledge Representation — Graph Databases
- Applied Machine Learning — Retrieval Augmented Generation
- Evaluation Metrics for AI Systems

### Learning Outcomes
- Designed and queried PostgreSQL schemas with advanced indexing
- Implemented MongoDB aggregation pipelines for experiment analytics
- Built Redis caching layer achieving 200x query speedup
- Set up ChromaDB and FAISS vector stores with hybrid search
- Built complete RAG pipeline from chunking to RAGAS evaluation
- Implemented Graph RAG with multi-hop Neo4j-style knowledge graph
- Applied 5 advanced RAG techniques including HyDE and cross-encoder reranking
- Built streaming QA app with multi-turn conversational memory

---

## Grand Capstone
DataSense AI — Autonomous Intelligent Data Analysis & Insights Platform
4 Agents | 6 MCP Servers | 40+ Tools | SQL + Power BI + ML Fusion
