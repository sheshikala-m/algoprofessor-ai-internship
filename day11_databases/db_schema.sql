-- ============================================
-- DATABASE SCHEMA FOR DATA SCIENCE PROJECT
-- Day 11: PostgreSQL + Async SQLAlchemy
-- Author: [Your Name]
-- Date: 2026-03-11
-- ============================================

-- Drop existing tables (clean setup)
DROP TABLE IF EXISTS vector_embeddings CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;
DROP TABLE IF EXISTS ml_projects CASCADE;
DROP TABLE IF EXISTS researchers CASCADE;

-- ============================================
-- TABLE 1: RESEARCHERS
-- ============================================
CREATE TABLE researchers (
    researcher_id   SERIAL PRIMARY KEY,
    username        VARCHAR(50)  NOT NULL UNIQUE,
    email           VARCHAR(100) NOT NULL UNIQUE,
    department      VARCHAR(100),
    expertise       TEXT[],                          -- e.g. ARRAY['NLP','CV']
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE researchers IS 'Data scientists / researchers who run experiments';
COMMENT ON COLUMN researchers.expertise IS 'Array of ML domain tags';

CREATE INDEX idx_researchers_email      ON researchers(email);
CREATE INDEX idx_researchers_department ON researchers(department);

-- ============================================
-- TABLE 2: ML_PROJECTS
-- ============================================
CREATE TABLE ml_projects (
    project_id      SERIAL PRIMARY KEY,
    project_name    VARCHAR(150) NOT NULL,
    description     TEXT,
    domain          VARCHAR(50)  NOT NULL,           -- NLP, CV, Tabular, etc.
    is_active       BOOLEAN      DEFAULT TRUE,
    created_by      INTEGER      REFERENCES researchers(researcher_id) ON DELETE SET NULL,
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE ml_projects IS 'Top-level ML research projects';

CREATE INDEX idx_projects_domain ON ml_projects(domain);
CREATE INDEX idx_projects_active ON ml_projects(is_active);

-- ============================================
-- TABLE 3: DATASETS
-- ============================================
CREATE TABLE datasets (
    dataset_id      SERIAL PRIMARY KEY,
    name            VARCHAR(150) NOT NULL,
    source          VARCHAR(200),                    -- Kaggle, HuggingFace, custom
    row_count       INTEGER,
    feature_count   INTEGER,
    task_type       VARCHAR(50),                     -- classification, regression, etc.
    metadata        JSONB,                           -- flexible extra info
    file_path       TEXT,
    project_id      INTEGER REFERENCES ml_projects(project_id) ON DELETE CASCADE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE datasets    IS 'Datasets used across experiments';
COMMENT ON COLUMN datasets.metadata IS 'Flexible JSONB: source URL, license, splits, etc.';

CREATE INDEX idx_datasets_project   ON datasets(project_id);
CREATE INDEX idx_datasets_task_type ON datasets(task_type);
CREATE INDEX idx_datasets_metadata  ON datasets USING GIN (metadata);  -- JSONB index

-- ============================================
-- TABLE 4: EXPERIMENTS
-- ============================================
CREATE TABLE experiments (
    experiment_id   SERIAL PRIMARY KEY,
    experiment_name VARCHAR(200) NOT NULL,
    project_id      INTEGER NOT NULL REFERENCES ml_projects(project_id) ON DELETE CASCADE,
    dataset_id      INTEGER REFERENCES datasets(dataset_id) ON DELETE SET NULL,
    researcher_id   INTEGER REFERENCES researchers(researcher_id) ON DELETE SET NULL,
    model_type      VARCHAR(100),                    -- RandomForest, BERT, CNN, etc.
    hyperparameters JSONB,                           -- all HP as JSON
    status          VARCHAR(20) DEFAULT 'pending'
                        CHECK (status IN ('pending','running','completed','failed')),
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE experiments IS 'Individual ML experiment runs';
COMMENT ON COLUMN experiments.hyperparameters IS 'Flexible JSONB: lr, epochs, batch_size, etc.';

CREATE INDEX idx_experiments_project    ON experiments(project_id);
CREATE INDEX idx_experiments_status     ON experiments(status);
CREATE INDEX idx_experiments_model      ON experiments(model_type);
CREATE INDEX idx_experiments_hparams    ON experiments USING GIN (hyperparameters);

-- ============================================
-- TABLE 5: MODEL_METRICS
-- ============================================
CREATE TABLE model_metrics (
    metric_id       SERIAL PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    split           VARCHAR(20) DEFAULT 'test'
                        CHECK (split IN ('train','validation','test')),
    accuracy        DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall          DECIMAL(6,4),
    f1_score        DECIMAL(6,4),
    loss            DECIMAL(10,6),
    extra_metrics   JSONB,                           -- AUC, BLEU, ROUGE, etc.
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE model_metrics IS 'Evaluation metrics per experiment per split';
COMMENT ON COLUMN model_metrics.extra_metrics IS 'Domain-specific metrics as JSONB';

CREATE INDEX idx_metrics_experiment ON model_metrics(experiment_id);
CREATE INDEX idx_metrics_split      ON model_metrics(split);
CREATE INDEX idx_metrics_f1         ON model_metrics(f1_score DESC);

-- ============================================
-- TABLE 6: VECTOR_EMBEDDINGS (for RAG)
-- ============================================
CREATE TABLE vector_embeddings (
    embedding_id    SERIAL PRIMARY KEY,
    dataset_id      INTEGER REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    chunk_text      TEXT NOT NULL,
    chunk_index     INTEGER,
    model_name      VARCHAR(100),                    -- e.g. text-embedding-3-small
    embedding_dim   INTEGER,
    metadata        JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE vector_embeddings IS 'Text chunks + embedding metadata for RAG pipeline';

CREATE INDEX idx_embeddings_dataset ON vector_embeddings(dataset_id);
CREATE INDEX idx_embeddings_model   ON vector_embeddings(model_name);

-- ============================================
-- VIEWS
-- ============================================

-- View 1: Experiment leaderboard (best F1 per project)
CREATE VIEW experiment_leaderboard AS
SELECT
    p.project_name,
    e.experiment_name,
    e.model_type,
    mm.split,
    mm.f1_score,
    mm.accuracy,
    mm.loss,
    r.username        AS researcher,
    e.created_at
FROM experiments e
JOIN ml_projects   p  ON e.project_id    = p.project_id
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
LEFT JOIN researchers r ON e.researcher_id = r.researcher_id
WHERE mm.split = 'test'
ORDER BY mm.f1_score DESC;

-- View 2: Dataset summary
CREATE VIEW dataset_summary AS
SELECT
    d.dataset_id,
    d.name,
    d.task_type,
    d.row_count,
    d.feature_count,
    p.project_name,
    COUNT(e.experiment_id) AS total_experiments,
    MAX(mm.f1_score)       AS best_f1
FROM datasets d
LEFT JOIN ml_projects  p  ON d.project_id    = p.project_id
LEFT JOIN experiments  e  ON d.dataset_id    = e.dataset_id
LEFT JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
GROUP BY d.dataset_id, d.name, d.task_type, d.row_count, d.feature_count, p.project_name;

-- View 3: Researcher activity
CREATE VIEW researcher_activity AS
SELECT
    r.username,
    r.department,
    COUNT(DISTINCT e.experiment_id)  AS total_experiments,
    COUNT(DISTINCT e.project_id)     AS total_projects,
    AVG(mm.f1_score)::DECIMAL(6,4)  AS avg_f1_score,
    MAX(e.created_at)                AS last_active
FROM researchers r
LEFT JOIN experiments  e  ON r.researcher_id = e.researcher_id
LEFT JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
GROUP BY r.researcher_id, r.username, r.department;

-- ============================================
-- STORED FUNCTIONS
-- ============================================

-- Function 1: Auto update updated_at
CREATE OR REPLACE FUNCTION fn_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function 2: Get best experiment for a project
CREATE OR REPLACE FUNCTION fn_best_experiment(p_project_id INTEGER)
RETURNS TABLE(
    experiment_name VARCHAR,
    model_type      VARCHAR,
    f1_score        DECIMAL,
    accuracy        DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.experiment_name,
        e.model_type,
        mm.f1_score,
        mm.accuracy
    FROM experiments e
    JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
    WHERE e.project_id = p_project_id
      AND mm.split = 'test'
    ORDER BY mm.f1_score DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function 3: Experiment completion duration (minutes)
CREATE OR REPLACE FUNCTION fn_experiment_duration(p_experiment_id INTEGER)
RETURNS DECIMAL AS $$
DECLARE
    v_minutes DECIMAL;
BEGIN
    SELECT EXTRACT(EPOCH FROM (completed_at - started_at)) / 60
    INTO v_minutes
    FROM experiments
    WHERE experiment_id = p_experiment_id;

    RETURN COALESCE(v_minutes, 0);
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- TRIGGERS
-- ============================================

CREATE TRIGGER trg_researchers_updated_at
BEFORE UPDATE ON researchers
FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

CREATE TRIGGER trg_projects_updated_at
BEFORE UPDATE ON ml_projects
FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

-- ============================================
-- VERIFICATION
-- ============================================
SELECT table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name;

SELECT table_name AS view_name
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;

DO $$
BEGIN
    RAISE NOTICE '=========================================';
    RAISE NOTICE 'Day 11 Schema Created Successfully!';
    RAISE NOTICE '=========================================';
    RAISE NOTICE 'Tables  : 6 (researchers, ml_projects, datasets, experiments, model_metrics, vector_embeddings)';
    RAISE NOTICE 'Views   : 3 (experiment_leaderboard, dataset_summary, researcher_activity)';
    RAISE NOTICE 'Functions: 3 (fn_update_timestamp, fn_best_experiment, fn_experiment_duration)';
    RAISE NOTICE 'Triggers : 2 (researchers, ml_projects updated_at)';
    RAISE NOTICE '=========================================';
END $$;
