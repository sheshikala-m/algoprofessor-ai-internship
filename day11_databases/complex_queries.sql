-- ============================================
-- COMPLEX SQL QUERIES - DATA SCIENCE PROJECT
-- Day 11: PostgreSQL + Async SQLAlchemy
-- ============================================

-- ============================================
-- SECTION 1: BASIC QUERIES (DQL)
-- ============================================

-- Query 1: All active projects
SELECT * FROM ml_projects WHERE is_active = TRUE ORDER BY created_at DESC;

-- Query 2: Filter experiments by status
SELECT
    experiment_name,
    model_type,
    status
FROM experiments
WHERE status = 'completed'
ORDER BY created_at DESC;

-- Query 3: Pattern matching on model type
SELECT
    experiment_id,
    experiment_name,
    model_type
FROM experiments
WHERE model_type ILIKE '%bert%'   -- case-insensitive BERT variants
ORDER BY created_at;

-- Query 4: Multiple conditions
SELECT
    name,
    task_type,
    row_count,
    feature_count
FROM datasets
WHERE task_type = 'classification'
    AND row_count > 100000
ORDER BY row_count DESC;

-- Query 5: IN operator
SELECT username, email, department
FROM researchers
WHERE department IN ('NLP Research', 'Data Science', 'Machine Learning');

-- Query 6: BETWEEN for metric ranges
SELECT
    experiment_id,
    split,
    f1_score,
    accuracy
FROM model_metrics
WHERE f1_score BETWEEN 0.90 AND 0.96
ORDER BY f1_score DESC;

-- ============================================
-- SECTION 2: AGGREGATE FUNCTIONS
-- ============================================

-- Query 7: Overall metrics summary
SELECT
    COUNT(*)              AS total_metric_rows,
    AVG(f1_score)         AS avg_f1,
    MAX(f1_score)         AS best_f1,
    MIN(loss)             AS best_loss,
    AVG(accuracy)         AS avg_accuracy
FROM model_metrics
WHERE split = 'test';

-- Query 8: Experiments per project
SELECT
    p.project_name,
    COUNT(e.experiment_id)  AS total_experiments,
    COUNT(CASE WHEN e.status = 'completed' THEN 1 END) AS completed,
    COUNT(CASE WHEN e.status = 'running'   THEN 1 END) AS running
FROM ml_projects p
LEFT JOIN experiments e ON p.project_id = e.project_id
GROUP BY p.project_id, p.project_name
ORDER BY total_experiments DESC;

-- Query 9: Average metrics grouped by model type
SELECT
    e.model_type,
    COUNT(*)              AS experiment_count,
    AVG(mm.f1_score)      AS avg_f1,
    AVG(mm.accuracy)      AS avg_accuracy,
    MIN(mm.loss)          AS best_loss
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
GROUP BY e.model_type
ORDER BY avg_f1 DESC;

-- Query 10: HAVING - only model types with multiple runs
SELECT
    model_type,
    COUNT(*) AS runs,
    AVG(f1_score) AS avg_f1
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
GROUP BY model_type
HAVING COUNT(*) >= 2
ORDER BY avg_f1 DESC;

-- ============================================
-- SECTION 3: JOINS
-- ============================================

-- Query 11: Full experiment details with researcher + project
SELECT
    p.project_name,
    r.username          AS researcher,
    e.experiment_name,
    e.model_type,
    e.status,
    e.created_at
FROM experiments e
INNER JOIN ml_projects  p ON e.project_id    = p.project_id
INNER JOIN researchers  r ON e.researcher_id = r.researcher_id
ORDER BY e.created_at DESC;

-- Query 12: Complete leaderboard: project, model, metrics
SELECT
    p.project_name,
    e.experiment_name,
    e.model_type,
    mm.split,
    mm.f1_score,
    mm.accuracy,
    mm.loss,
    r.username AS researcher
FROM experiments e
INNER JOIN ml_projects   p  ON e.project_id    = p.project_id
INNER JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
INNER JOIN researchers   r  ON e.researcher_id = r.researcher_id
WHERE mm.split = 'test'
ORDER BY mm.f1_score DESC;

-- Query 13: LEFT JOIN - datasets with no experiments yet
SELECT
    d.dataset_id,
    d.name,
    d.task_type,
    d.row_count
FROM datasets d
LEFT JOIN experiments e ON d.dataset_id = e.dataset_id
WHERE e.experiment_id IS NULL;

-- Query 14: All researchers with their experiment count
SELECT
    r.username,
    r.department,
    COUNT(e.experiment_id)          AS total_experiments,
    COALESCE(AVG(mm.f1_score), 0)  AS avg_f1_achieved
FROM researchers r
LEFT JOIN experiments   e  ON r.researcher_id = e.researcher_id
LEFT JOIN model_metrics mm ON e.experiment_id = mm.experiment_id AND mm.split = 'test'
GROUP BY r.researcher_id, r.username, r.department
ORDER BY total_experiments DESC;

-- Query 15: SELF JOIN - experiments on same dataset
SELECT
    e1.experiment_name AS experiment_a,
    e2.experiment_name AS experiment_b,
    e1.model_type      AS model_a,
    e2.model_type      AS model_b,
    d.name             AS dataset
FROM experiments e1
INNER JOIN experiments e2 ON e1.dataset_id = e2.dataset_id
    AND e1.experiment_id < e2.experiment_id
INNER JOIN datasets d ON e1.dataset_id = d.dataset_id
ORDER BY dataset;

-- ============================================
-- SECTION 4: SUBQUERIES
-- ============================================

-- Query 16: Experiments above average F1
SELECT
    e.experiment_name,
    e.model_type,
    mm.f1_score
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
  AND mm.f1_score > (
      SELECT AVG(f1_score)
      FROM model_metrics
      WHERE split = 'test'
  )
ORDER BY mm.f1_score DESC;

-- Query 17: Researchers who ran experiments with F1 > 0.93
SELECT username, email
FROM researchers
WHERE researcher_id IN (
    SELECT e.researcher_id
    FROM experiments e
    JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
    WHERE mm.split = 'test' AND mm.f1_score > 0.93
);

-- Query 18: Correlated subquery - each experiment vs project average
SELECT
    e.experiment_name,
    e.model_type,
    mm.f1_score,
    (SELECT AVG(mm2.f1_score)
     FROM experiments e2
     JOIN model_metrics mm2 ON e2.experiment_id = mm2.experiment_id
     WHERE e2.project_id = e.project_id AND mm2.split = 'test'
    ) AS project_avg_f1
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY e.project_id, mm.f1_score DESC;

-- Query 19: Derived table - projects with avg F1 > 0.90
SELECT project_name, avg_f1, experiment_count
FROM (
    SELECT
        p.project_name,
        AVG(mm.f1_score)         AS avg_f1,
        COUNT(e.experiment_id)   AS experiment_count
    FROM ml_projects p
    JOIN experiments   e  ON p.project_id    = e.project_id
    JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
    WHERE mm.split = 'test'
    GROUP BY p.project_id, p.project_name
) AS project_stats
WHERE avg_f1 > 0.90
ORDER BY avg_f1 DESC;

-- ============================================
-- SECTION 5: CTEs
-- ============================================

-- Query 20: Best experiment per project
WITH best_per_project AS (
    SELECT
        e.project_id,
        e.experiment_name,
        e.model_type,
        mm.f1_score,
        ROW_NUMBER() OVER (PARTITION BY e.project_id ORDER BY mm.f1_score DESC) AS rn
    FROM experiments e
    JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
    WHERE mm.split = 'test'
)
SELECT
    p.project_name,
    b.experiment_name,
    b.model_type,
    b.f1_score AS best_f1
FROM best_per_project b
JOIN ml_projects p ON b.project_id = p.project_id
WHERE rn = 1
ORDER BY best_f1 DESC;

-- Query 21: Multiple CTEs - high performing researchers + recent work
WITH
top_researchers AS (
    SELECT
        r.researcher_id,
        r.username,
        AVG(mm.f1_score) AS avg_f1
    FROM researchers r
    JOIN experiments   e  ON r.researcher_id = e.researcher_id
    JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
    WHERE mm.split = 'test'
    GROUP BY r.researcher_id, r.username
    HAVING AVG(mm.f1_score) > 0.90
),
recent_experiments AS (
    SELECT researcher_id, COUNT(*) AS recent_runs
    FROM experiments
    WHERE created_at >= CURRENT_DATE - INTERVAL '14 days'
    GROUP BY researcher_id
)
SELECT
    tr.username,
    tr.avg_f1,
    COALESCE(re.recent_runs, 0) AS recent_runs
FROM top_researchers tr
LEFT JOIN recent_experiments re ON tr.researcher_id = re.researcher_id
ORDER BY tr.avg_f1 DESC;

-- Query 22: CTE with UPDATE + RETURNING
WITH deactivated AS (
    UPDATE ml_projects
    SET is_active = FALSE
    WHERE created_at < CURRENT_DATE - INTERVAL '90 days'
    RETURNING project_id, project_name
)
SELECT 'Projects Deactivated' AS action, COUNT(*) AS count
FROM deactivated;

-- ============================================
-- SECTION 6: RECURSIVE CTEs
-- ============================================

-- Query 23: Build NLP experiment chain (conceptual lineage)
WITH RECURSIVE experiment_chain AS (
    -- Base: first experiment in project 1
    SELECT
        experiment_id,
        experiment_name,
        model_type,
        1 AS generation,
        experiment_name::TEXT AS lineage
    FROM experiments
    WHERE project_id = 1
    ORDER BY experiment_id
    LIMIT 1

    UNION ALL

    SELECT
        e.experiment_id,
        e.experiment_name,
        e.model_type,
        ec.generation + 1,
        ec.lineage || ' → ' || e.experiment_name
    FROM experiments e
    INNER JOIN experiment_chain ec ON e.experiment_id = ec.experiment_id + 1
        AND e.project_id = 1
    WHERE ec.generation < 5
)
SELECT
    REPEAT('  ', generation - 1) || experiment_name AS hierarchy,
    model_type,
    lineage
FROM experiment_chain
ORDER BY generation;

-- Query 24: Generate date series for experiment timeline
WITH RECURSIVE date_range AS (
    SELECT DATE '2026-03-01' AS run_date
    UNION ALL
    SELECT run_date + INTERVAL '1 day'
    FROM date_range
    WHERE run_date < DATE '2026-03-11'
)
SELECT
    dr.run_date,
    COUNT(e.experiment_id)       AS experiments_started,
    COUNT(mm.metric_id)          AS metrics_logged
FROM date_range dr
LEFT JOIN experiments   e  ON DATE(e.started_at) = dr.run_date
LEFT JOIN model_metrics mm ON DATE(mm.recorded_at) = dr.run_date
GROUP BY dr.run_date
ORDER BY dr.run_date;

-- ============================================
-- SECTION 7: WINDOW FUNCTIONS
-- ============================================

-- Query 25: ROW_NUMBER - rank experiments by F1 per project
SELECT
    p.project_name,
    e.experiment_name,
    e.model_type,
    mm.f1_score,
    ROW_NUMBER() OVER (PARTITION BY e.project_id ORDER BY mm.f1_score DESC) AS rank_in_project
FROM experiments e
JOIN ml_projects   p  ON e.project_id    = p.project_id
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY p.project_name, rank_in_project;

-- Query 26: RANK vs DENSE_RANK on F1 scores
SELECT
    experiment_name,
    model_type,
    f1_score,
    RANK()       OVER (ORDER BY f1_score DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY f1_score DESC) AS dense_rank
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY rank;

-- Query 27: LAG/LEAD - compare consecutive experiment F1
SELECT
    e.experiment_name,
    mm.f1_score,
    LAG(mm.f1_score)  OVER (ORDER BY e.experiment_id) AS prev_f1,
    LEAD(mm.f1_score) OVER (ORDER BY e.experiment_id) AS next_f1,
    mm.f1_score - LAG(mm.f1_score) OVER (ORDER BY e.experiment_id) AS f1_improvement
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY e.experiment_id;

-- Query 28: Running average F1 over time
SELECT
    e.created_at::DATE                                         AS run_date,
    e.experiment_name,
    mm.f1_score,
    AVG(mm.f1_score) OVER (
        ORDER BY e.created_at
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )                                                          AS rolling_avg_f1,
    SUM(mm.f1_score) OVER (ORDER BY e.created_at)             AS cumulative_f1_sum
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY e.created_at;

-- Query 29: NTILE - tier experiments by performance
SELECT
    experiment_name,
    model_type,
    f1_score,
    NTILE(4) OVER (ORDER BY f1_score DESC) AS performance_quartile,
    CASE NTILE(4) OVER (ORDER BY f1_score DESC)
        WHEN 1 THEN 'State-of-the-Art'
        WHEN 2 THEN 'Strong'
        WHEN 3 THEN 'Moderate'
        WHEN 4 THEN 'Needs Improvement'
    END AS performance_tier
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY f1_score DESC;

-- Query 30: FIRST_VALUE / LAST_VALUE - best and worst per project
SELECT
    p.project_name,
    e.experiment_name,
    mm.f1_score,
    FIRST_VALUE(e.experiment_name) OVER (
        PARTITION BY e.project_id ORDER BY mm.f1_score DESC
    ) AS best_model,
    LAST_VALUE(e.experiment_name) OVER (
        PARTITION BY e.project_id ORDER BY mm.f1_score DESC
        RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS worst_model
FROM experiments e
JOIN ml_projects   p  ON e.project_id    = p.project_id
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
ORDER BY p.project_name, mm.f1_score DESC;

-- ============================================
-- SECTION 8: ADVANCED ANALYTICS
-- ============================================

-- Query 31: Model comparison across metrics (pivot-style)
SELECT
    e.model_type,
    COUNT(*)                     AS num_experiments,
    ROUND(AVG(mm.f1_score)::NUMERIC, 4)   AS avg_f1,
    ROUND(AVG(mm.accuracy)::NUMERIC, 4)   AS avg_accuracy,
    ROUND(AVG(mm.loss)::NUMERIC, 6)       AS avg_loss,
    ROUND(MAX(mm.f1_score)::NUMERIC, 4)   AS peak_f1,
    ROUND(MIN(mm.loss)::NUMERIC, 6)       AS best_loss
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
GROUP BY e.model_type
ORDER BY avg_f1 DESC;

-- Query 32: Dataset efficiency (F1 per 100K rows)
SELECT
    d.name                              AS dataset,
    d.task_type,
    d.row_count,
    AVG(mm.f1_score)                   AS avg_f1,
    AVG(mm.f1_score) / (d.row_count / 100000.0) AS f1_per_100k_rows
FROM datasets d
JOIN experiments   e  ON d.dataset_id    = e.dataset_id
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test' AND d.row_count IS NOT NULL
GROUP BY d.dataset_id, d.name, d.task_type, d.row_count
ORDER BY f1_per_100k_rows DESC;

-- Query 33: Researcher performance summary (RFM-style)
WITH researcher_stats AS (
    SELECT
        r.researcher_id,
        r.username,
        r.department,
        MAX(e.created_at)      AS last_experiment,
        COUNT(e.experiment_id) AS total_experiments,
        AVG(mm.f1_score)       AS avg_f1
    FROM researchers r
    JOIN experiments   e  ON r.researcher_id = e.researcher_id
    JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
    WHERE mm.split = 'test'
    GROUP BY r.researcher_id, r.username, r.department
)
SELECT
    username,
    department,
    last_experiment,
    total_experiments,
    ROUND(avg_f1::NUMERIC, 4)  AS avg_f1,
    NTILE(3) OVER (ORDER BY last_experiment DESC)  AS recency_score,
    NTILE(3) OVER (ORDER BY total_experiments)     AS volume_score,
    NTILE(3) OVER (ORDER BY avg_f1)               AS quality_score
FROM researcher_stats
ORDER BY quality_score DESC;

-- Query 34: Train vs Validation vs Test gap (overfitting check)
SELECT
    e.experiment_name,
    e.model_type,
    MAX(CASE WHEN mm.split = 'train'      THEN mm.f1_score END) AS train_f1,
    MAX(CASE WHEN mm.split = 'validation' THEN mm.f1_score END) AS val_f1,
    MAX(CASE WHEN mm.split = 'test'       THEN mm.f1_score END) AS test_f1,
    MAX(CASE WHEN mm.split = 'train'      THEN mm.f1_score END)
    - MAX(CASE WHEN mm.split = 'test'     THEN mm.f1_score END) AS overfit_gap
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
GROUP BY e.experiment_id, e.experiment_name, e.model_type
HAVING MAX(CASE WHEN mm.split = 'train' THEN mm.f1_score END) IS NOT NULL
ORDER BY overfit_gap DESC;

-- ============================================
-- SUCCESS MESSAGE
-- ============================================
SELECT 'All 34 complex queries executed successfully!' AS status;
SELECT 'Concepts: DQL, Aggregates, JOINs, Subqueries, CTEs, Recursive CTEs, Window Functions, Analytics' AS summary;
