# Day 11: PostgreSQL Databases
Author: Sheshikala
Date: March 11, 2026

## What I did 
Learned PostgreSQL and SQL from basics to advanced. I used an ML Experiment Tracker as my project theme instead of the usual ecommerce example because it felt more relevant to what we are actually doing in this internship.

Took me a while to set up pgAdmin properly and connect to the database. The partitioning section was the hardest part.

## Files

db_schema.sql - all 6 tables with constraints and foreign keys
sample_data.sql - inserted realistic ML experiment data
complex_queries.sql - 34 queries from basic SELECT to window functions
jsonb_examples.sql - JSONB operators and GIN indexes
advanced_indexing.sql - B-Tree, GIN, GiST, BRIN, partial indexes
partitioning.sql - range, list, hash partitioning
verification_queries.sql - run this last to check everything works
migrations/001_initial.sql - creates all tables
migrations/002_add_indexes.sql - adds indexes

## How to run

1. Open pgAdmin and connect to PostgreSQL
2. Create a new database called ml_tracker
3. Open query tool and run files in this order:
   - db_schema.sql
   - sample_data.sql
   - complex_queries.sql
   - jsonb_examples.sql
   - advanced_indexing.sql
   - partitioning.sql
   - verification_queries.sql

## What I learned

JSONB is really useful for storing hyperparameters because BERT and XGBoost have completely different parameters. Using JSONB means I dont need separate columns for each model type.

Window functions were confusing at first. RANK() vs DENSE_RANK() kept tripping me up - DENSE doesnt skip numbers after a tie but RANK does.

Recursive CTEs took longest to understand. Had to read it 3 times before it made sense. Its basically like recursion in Python - needs a base case and recursive case.

BRIN index only works on naturally ordered data like timestamps. I tried it on a random column first and it didnt help at all.

## What was difficult

- Kept forgetting UNBOUNDED FOLLOWING in LAST_VALUE window function
- $lookup vs JOIN syntax difference
- Partition pruning - had to use EXPLAIN ANALYZE to verify it was actually working
- JSONB containment operator @> syntax was not obvious

## Tables I created

researchers - data scientists with expertise array
ml_projects - top level projects
datasets - training data with metadata in JSONB
experiments - model runs with hyperparameters in JSONB
model_metrics - f1, accuracy, loss per experiment
vector_embeddings - text chunks for later RAG work
