-- ============================================
-- MIGRATION 001: INITIAL SCHEMA
-- Day 11: PostgreSQL Databases
-- Description: Create all base tables
-- Date: 2026-03-11
-- ============================================

-- ============================================
-- STEP 1: CREATE TABLES
-- ============================================

CREATE TABLE IF NOT EXISTS researchers (
    researcher_id   SERIAL PRIMARY KEY,
    username        VARCHAR(50)  NOT NULL UNIQUE,
    email           VARCHAR(100) NOT NULL UNIQUE,
    department      VARCHAR(100),
    expertise       TEXT[],
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_projects (
    project_id      SERIAL PRIMARY KEY,
    project_name    VARCHAR(150) NOT NULL,
    description     TEXT,
    domain          VARCHAR(50)  NOT NULL,
    is_active       BOOLEAN   DEFAULT TRUE,
    created_by      INTEGER   REFERENCES researchers(researcher_id) ON DELETE SET NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id      SERIAL PRIMARY KEY,
    name            VARCHAR(150) NOT NULL,
    source          VARCHAR(200),
    row_count       INTEGER,
    feature_count   INTEGER,
    task_type       VARCHAR(50),
    metadata        JSONB,
    file_path       TEXT,
    project_id      INTEGER REFERENCES ml_projects(project_id) ON DELETE CASCADE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   SERIAL PRIMARY KEY,
    experiment_name VARCHAR(200) NOT NULL,
    project_id      INTEGER NOT NULL REFERENCES ml_projects(project_id) ON DELETE CASCADE,
    dataset_id      INTEGER REFERENCES datasets(dataset_id) ON DELETE SET NULL,
    researcher_id   INTEGER REFERENCES researchers(researcher_id) ON DELETE SET NULL,
    model_type      VARCHAR(100),
    hyperparameters JSONB,
    status          VARCHAR(20) DEFAULT 'pending'
                        CHECK (status IN ('pending','running','completed','failed')),
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id       SERIAL PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    split           VARCHAR(20) DEFAULT 'test'
                        CHECK (split IN ('train','validation','test')),
    accuracy        DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall          DECIMAL(6,4),
    f1_score        DECIMAL(6,4),
    loss            DECIMAL(10,6),
    extra_metrics   JSONB,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vector_embeddings (
    embedding_id    SERIAL PRIMARY KEY,
    dataset_id      INTEGER REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    chunk_text      TEXT NOT NULL,
    chunk_index     INTEGER,
    model_name      VARCHAR(100),
    embedding_dim   INTEGER,
    metadata        JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- STEP 2: CREATE VIEWS
-- ============================================

CREATE OR REPLACE VIEW experiment_leaderboard AS
SELECT
    p.project_name,
    e.experiment_name,
    e.model_type,
    mm.split,
    mm.f1_score,
    mm.accuracy,
    r.username AS researcher
FROM experiments e
JOIN ml_projects   p  ON e.project_id    = p.project_id
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
LEFT JOIN researchers r ON e.researcher_id = r.researcher_id
WHERE mm.split = 'test'
ORDER BY mm.f1_score DESC;

CREATE OR REPLACE VIEW dataset_summary AS
SELECT
    d.dataset_id,
    d.name,
    d.task_type,
    d.row_count,
    p.project_name,
    COUNT(e.experiment_id) AS total_experiments,
    MAX(mm.f1_score)       AS best_f1
FROM datasets d
LEFT JOIN ml_projects   p  ON d.project_id    = p.project_id
LEFT JOIN experiments   e  ON d.dataset_id    = e.dataset_id
LEFT JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
GROUP BY d.dataset_id, d.name, d.task_type, d.row_count, p.project_name;

CREATE OR REPLACE VIEW researcher_activity AS
SELECT
    r.username,
    r.department,
    COUNT(DISTINCT e.experiment_id) AS total_experiments,
    AVG(mm.f1_score)::DECIMAL(6,4) AS avg_f1_score
FROM researchers r
LEFT JOIN experiments   e  ON r.researcher_id = e.researcher_id
LEFT JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
GROUP BY r.researcher_id, r.username, r.department;

-- ============================================
-- STEP 3: CREATE FUNCTIONS + TRIGGERS
-- ============================================

CREATE OR REPLACE FUNCTION fn_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_researchers_updated_at
BEFORE UPDATE ON researchers
FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

CREATE OR REPLACE TRIGGER trg_projects_updated_at
BEFORE UPDATE ON ml_projects
FOR EACH ROW EXECUTE FUNCTION fn_update_timestamp();

-- ============================================
-- VERIFY
-- ============================================
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name;

SELECT 'Migration 001 — Initial schema applied successfully!' AS status;
