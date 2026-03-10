-- ============================================
-- ALL POSTGRESQL INDEX TYPES
-- Day 11: Data Science Project Context
-- ============================================

-- ============================================
-- PART 1: B-TREE INDEX (DEFAULT)
-- ============================================
-- Best for: =, <, >, BETWEEN, ORDER BY
-- Already created in db_schema.sql for most cols
-- Creating additional ones here for demo

CREATE INDEX IF NOT EXISTS idx_metrics_f1_btree
ON model_metrics(f1_score DESC);

CREATE INDEX IF NOT EXISTS idx_metrics_loss_btree
ON model_metrics(loss ASC);

-- Test B-Tree: range query on F1 score
SELECT experiment_id, split, f1_score, accuracy
FROM model_metrics
WHERE f1_score BETWEEN 0.90 AND 0.95
ORDER BY f1_score DESC;

EXPLAIN ANALYZE
SELECT * FROM model_metrics
WHERE f1_score > 0.93;

-- ============================================
-- PART 2: HASH INDEX
-- ============================================
-- Best for: exact equality (=) only

CREATE INDEX IF NOT EXISTS idx_experiments_status_hash
ON experiments USING HASH (status);

CREATE INDEX IF NOT EXISTS idx_experiments_model_hash
ON experiments USING HASH (model_type);

-- Test Hash: exact match query
SELECT experiment_name, model_type, created_at
FROM experiments
WHERE status = 'completed';

EXPLAIN ANALYZE
SELECT * FROM experiments
WHERE model_type = 'BERT';

-- ============================================
-- PART 3: GIN INDEX — JSONB
-- ============================================
-- Best for: JSONB containment @>, ?
-- Already created in db_schema.sql

-- Test GIN on hyperparameters JSONB
SELECT experiment_name, hyperparameters
FROM experiments
WHERE hyperparameters @> '{"pretrained": true}';

-- Test GIN on dataset metadata
SELECT name, metadata
FROM datasets
WHERE metadata @> '{"license": "CC0"}';

-- Test GIN: key existence
SELECT experiment_name
FROM experiments
WHERE hyperparameters ? 'warmup_steps';

-- ============================================
-- PART 4: GIN INDEX — ARRAY
-- ============================================
-- Best for: array containment @>

CREATE INDEX IF NOT EXISTS idx_researchers_expertise_gin
ON researchers USING GIN (expertise);

-- Test GIN array: find NLP researchers
SELECT username, department, expertise
FROM researchers
WHERE expertise @> ARRAY['NLP'];

-- Find researchers with both NLP and Transformers
SELECT username, expertise
FROM researchers
WHERE expertise @> ARRAY['NLP', 'Transformers'];

-- ============================================
-- PART 5: GiST INDEX
-- ============================================
-- Best for: geometric/range/full-text data

-- Full-text search index on chunk text (RAG)
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_gist
ON vector_embeddings USING GiST (to_tsvector('english', chunk_text));

-- Test GiST: full-text search on embeddings
SELECT chunk_text, metadata
FROM vector_embeddings
WHERE to_tsvector('english', chunk_text) @@ to_tsquery('revenue & Q3');

-- Test GiST: search for AI mentions
SELECT chunk_text
FROM vector_embeddings
WHERE to_tsvector('english', chunk_text) @@ to_tsquery('AI | machine');

-- ============================================
-- PART 6: BRIN INDEX — LARGE TABLES
-- ============================================
-- Best for: very large tables with sequential data (dates)

CREATE INDEX IF NOT EXISTS idx_metrics_recorded_brin
ON model_metrics USING BRIN (recorded_at);

CREATE INDEX IF NOT EXISTS idx_experiments_started_brin
ON experiments USING BRIN (started_at);

-- Test BRIN: date range query
SELECT experiment_id, split, f1_score, recorded_at
FROM model_metrics
WHERE recorded_at >= CURRENT_DATE - INTERVAL '7 days';

EXPLAIN ANALYZE
SELECT * FROM experiments
WHERE started_at >= '2026-03-01';

-- ============================================
-- PART 7: PARTIAL INDEX
-- ============================================
-- Index only rows matching a condition
-- Smaller + faster for filtered queries

-- Only index completed experiments
CREATE INDEX IF NOT EXISTS idx_experiments_completed_partial
ON experiments(created_at DESC)
WHERE status = 'completed';

-- Only index test-split metrics
CREATE INDEX IF NOT EXISTS idx_metrics_test_split_partial
ON model_metrics(f1_score DESC)
WHERE split = 'test';

-- Test partial index
SELECT experiment_name, model_type
FROM experiments
WHERE status = 'completed'
ORDER BY created_at DESC;

EXPLAIN ANALYZE
SELECT * FROM model_metrics
WHERE split = 'test'
  AND f1_score > 0.92;

-- ============================================
-- PART 8: COMPOSITE INDEX
-- ============================================
-- Index on multiple columns together

CREATE INDEX IF NOT EXISTS idx_experiments_project_status
ON experiments(project_id, status);

CREATE INDEX IF NOT EXISTS idx_metrics_exp_split
ON model_metrics(experiment_id, split);

-- Test composite: project + status filter
SELECT experiment_name, model_type
FROM experiments
WHERE project_id = 1 AND status = 'completed';

-- Test composite: experiment + split lookup
SELECT f1_score, accuracy, loss
FROM model_metrics
WHERE experiment_id = 2 AND split = 'test';

-- ============================================
-- PART 9: VIEW ALL INDEXES
-- ============================================
SELECT
    indexname       AS index_name,
    tablename       AS table_name,
    indexdef        AS definition
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- ============================================
-- PART 10: CHECK INDEX USAGE
-- ============================================
EXPLAIN ANALYZE
SELECT * FROM experiments
WHERE model_type = 'RoBERTa';

EXPLAIN ANALYZE
SELECT * FROM model_metrics
WHERE split = 'test' AND f1_score > 0.93;

EXPLAIN ANALYZE
SELECT * FROM researchers
WHERE expertise @> ARRAY['NLP'];

-- ============================================
-- SUMMARY
-- ============================================
SELECT 'All index types created and tested!' AS message;
SELECT 'Indexes: B-Tree, Hash, GIN (JSONB+Array), GiST (Full-Text), BRIN, Partial, Composite' AS summary;
