-- ============================================
-- JSONB IN ML / DATA SCIENCE CONTEXT
-- Day 11: PostgreSQL + Async SQLAlchemy
-- ============================================
-- JSONB stores flexible JSON — perfect for:
--   hyperparameters, model configs, dataset metadata
-- ============================================

-- ============================================
-- PART 1: WHY JSONB IN ML PROJECTS?
-- ============================================
-- Different models have different hyperparameters
-- Can't create fixed columns for all of them
-- JSONB lets each experiment store its own HP schema

-- ============================================
-- PART 2: QUERY HYPERPARAMETERS WITH ->
-- ============================================

-- Get learning rate for all experiments
SELECT
    experiment_name,
    model_type,
    hyperparameters->>'lr'         AS learning_rate,
    hyperparameters->>'epochs'     AS epochs,
    hyperparameters->>'batch_size' AS batch_size
FROM experiments
WHERE hyperparameters->>'lr' IS NOT NULL
ORDER BY (hyperparameters->>'lr')::FLOAT ASC;

-- ============================================
-- PART 3: FILTER BY JSON VALUE
-- ============================================

-- Find all experiments using AdamW optimizer
SELECT
    experiment_name,
    model_type,
    hyperparameters->>'optimizer' AS optimizer
FROM experiments
WHERE hyperparameters->>'optimizer' = 'AdamW';

-- Find experiments with large batch size
SELECT
    experiment_name,
    hyperparameters->>'batch_size' AS batch_size
FROM experiments
WHERE (hyperparameters->>'batch_size')::INTEGER >= 32
ORDER BY (hyperparameters->>'batch_size')::INTEGER DESC;

-- ============================================
-- PART 4: @> CONTAINMENT OPERATOR
-- ============================================

-- Find experiments that used pretrained weights
SELECT
    experiment_name,
    model_type,
    hyperparameters
FROM experiments
WHERE hyperparameters @> '{"pretrained": true}';

-- Find experiments with augmentation enabled
SELECT
    experiment_name,
    model_type
FROM experiments
WHERE hyperparameters @> '{"augmentation": true}';

-- ============================================
-- PART 5: QUERY DATASET METADATA
-- ============================================

-- Get license info from dataset metadata
SELECT
    name,
    metadata->>'license'   AS license,
    metadata->>'url'       AS source_url
FROM datasets
WHERE metadata->>'license' IS NOT NULL;

-- Query nested JSON — train/val/test splits
SELECT
    name,
    metadata->'splits'->>'train' AS train_split,
    metadata->'splits'->>'val'   AS val_split,
    metadata->'splits'->>'test'  AS test_split
FROM datasets
WHERE metadata ? 'splits';

-- ============================================
-- PART 6: QUERY EXTRA_METRICS (JSONB)
-- ============================================

-- Get AUC from NLP experiments
SELECT
    e.experiment_name,
    e.model_type,
    mm.f1_score,
    mm.extra_metrics->>'auc' AS auc_score
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.extra_metrics ? 'auc'
  AND mm.split = 'test'
ORDER BY (mm.extra_metrics->>'auc')::FLOAT DESC;

-- Get RAG-specific metrics
SELECT
    e.experiment_name,
    mm.extra_metrics->>'hit_rate'          AS hit_rate,
    mm.extra_metrics->>'mrr'               AS mrr,
    mm.extra_metrics->>'faithfulness'      AS faithfulness,
    mm.extra_metrics->>'answer_relevancy'  AS answer_relevancy
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.extra_metrics ? 'hit_rate';

-- ============================================
-- PART 7: ADD NEW HYPERPARAMETER TO JSON
-- ============================================

-- Add weight_decay to BERT experiment
UPDATE experiments
SET hyperparameters = hyperparameters || '{"weight_decay": 0.01}'
WHERE model_type = 'BERT';

-- Verify
SELECT experiment_name, hyperparameters->>'weight_decay' AS weight_decay
FROM experiments
WHERE model_type = 'BERT';

-- ============================================
-- PART 8: UPDATE A JSON VALUE
-- ============================================

-- Increase epochs for DistilBERT using jsonb_set
UPDATE experiments
SET hyperparameters = jsonb_set(
    hyperparameters,
    '{epochs}',
    '6'
)
WHERE model_type = 'DistilBERT';

-- Verify
SELECT
    experiment_name,
    hyperparameters->>'epochs' AS updated_epochs
FROM experiments
WHERE model_type = 'DistilBERT';

-- ============================================
-- PART 9: REMOVE A KEY FROM JSON
-- ============================================

-- Remove 'warmup_steps' from an experiment
UPDATE experiments
SET hyperparameters = hyperparameters - 'warmup_steps'
WHERE model_type = 'RoBERTa';

-- Verify key is gone
SELECT
    experiment_name,
    hyperparameters ? 'warmup_steps' AS has_warmup
FROM experiments
WHERE model_type = 'RoBERTa';

-- ============================================
-- PART 10: GIN INDEX ON JSONB
-- ============================================

-- Already created in db_schema.sql:
-- CREATE INDEX idx_experiments_hparams ON experiments USING GIN (hyperparameters);
-- CREATE INDEX idx_datasets_metadata   ON datasets    USING GIN (metadata);

-- Check index is being used
EXPLAIN ANALYZE
SELECT * FROM experiments
WHERE hyperparameters @> '{"pretrained": true}';

-- ============================================
-- PART 11: jsonb_each — EXPAND JSON TO ROWS
-- ============================================

-- Unpack all hyperparameters of an experiment as key-value rows
SELECT
    e.experiment_name,
    kv.key,
    kv.value
FROM experiments e,
     jsonb_each(e.hyperparameters) AS kv
WHERE e.experiment_id = 1;

-- ============================================
-- PART 12: jsonb_object_keys — LIST ALL KEYS
-- ============================================

-- What hyperparameter keys exist across all experiments?
SELECT DISTINCT jsonb_object_keys(hyperparameters) AS hp_key
FROM experiments
ORDER BY hp_key;

-- ============================================
-- PART 13: AGGREGATE JSON — BUILD SUMMARY
-- ============================================

-- Average learning rate per model type (from JSONB)
SELECT
    model_type,
    COUNT(*)                                              AS run_count,
    AVG((hyperparameters->>'lr')::FLOAT)                 AS avg_lr,
    AVG((hyperparameters->>'epochs')::INTEGER)           AS avg_epochs,
    AVG((hyperparameters->>'batch_size')::INTEGER)       AS avg_batch_size
FROM experiments
WHERE hyperparameters ? 'lr'
GROUP BY model_type
ORDER BY avg_lr;

-- ============================================
-- SUMMARY
-- ============================================
SELECT 'JSONB learning for ML projects completed!' AS message;
SELECT 'Operations covered: ->, ->>, @>, jsonb_set, jsonb_each, GIN index, aggregate on JSON' AS summary;
