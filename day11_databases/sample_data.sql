-- ============================================
-- SAMPLE DATA FOR DATA SCIENCE PROJECT
-- Day 11: PostgreSQL + Async SQLAlchemy
-- ============================================

-- ============================================
-- RESEARCHERS
-- ============================================
INSERT INTO researchers (username, email, department, expertise) VALUES
('arjun_ds',    'arjun@lab.com',    'Data Science',      ARRAY['NLP','LLMs']),
('priya_ml',    'priya@lab.com',    'Machine Learning',  ARRAY['CV','CNNs']),
('ravi_nlp',    'ravi@lab.com',     'NLP Research',      ARRAY['NLP','Transformers']),
('sneha_cv',    'sneha@lab.com',    'Computer Vision',   ARRAY['CV','ObjectDetection']),
('kiran_de',    'kiran@lab.com',    'Data Engineering',  ARRAY['Tabular','Feature Engineering']),
('meera_stat',  'meera@lab.com',    'Statistics',        ARRAY['Regression','Bayesian']),
('rohan_rl',    'rohan@lab.com',    'Reinforcement Learning', ARRAY['RL','DQN']),
('divya_mlops', 'divya@lab.com',    'MLOps',             ARRAY['MLOps','Pipelines']);

-- ============================================
-- ML PROJECTS
-- ============================================
INSERT INTO ml_projects (project_name, description, domain, created_by) VALUES
('Sentiment Analysis Engine',     'Customer review classification using transformers',   'NLP',      1),
('Medical Image Classifier',      'Disease detection from X-ray and MRI scans',          'CV',       2),
('Sales Forecasting System',      'Time series forecasting for retail sales',             'Tabular',  5),
('Document RAG Pipeline',         'Retrieval-Augmented Generation for knowledge base',    'NLP',      3),
('Fraud Detection System',        'Real-time fraud classification on transactions',       'Tabular',  6),
('Object Detection Benchmark',    'YOLO vs Faster-RCNN comparison on custom dataset',    'CV',       4);

-- ============================================
-- DATASETS
-- ============================================
INSERT INTO datasets (name, source, row_count, feature_count, task_type, metadata, project_id) VALUES
(
    'Amazon Reviews 2024',
    'HuggingFace',
    500000, 8, 'classification',
    '{"url":"hf.co/datasets/amazon","license":"CC-BY","splits":{"train":0.8,"val":0.1,"test":0.1},"language":"en"}',
    1
),
(
    'NIH Chest X-Ray',
    'Kaggle',
    112000, 15, 'classification',
    '{"url":"kaggle.com/nih-chest","license":"CC0","image_size":"224x224","classes":14}',
    2
),
(
    'Walmart Sales Dataset',
    'Kaggle',
    421000, 12, 'regression',
    '{"url":"kaggle.com/walmart","frequency":"weekly","stores":45,"departments":81}',
    3
),
(
    'Internal KB Documents',
    'custom',
    15000, 5, 'retrieval',
    '{"format":"PDF+TXT","avg_tokens":512,"domain":"finance","indexed":true}',
    4
),
(
    'Credit Card Transactions',
    'Kaggle',
    284807, 30, 'classification',
    '{"url":"kaggle.com/creditcard","fraud_ratio":0.00172,"balanced":false}',
    5
),
(
    'COCO Custom Subset',
    'custom',
    28000, 20, 'detection',
    '{"classes":10,"format":"YOLO","image_size":"640x640","augmented":true}',
    6
);

-- ============================================
-- EXPERIMENTS
-- ============================================
INSERT INTO experiments
    (experiment_name, project_id, dataset_id, researcher_id, model_type,
     hyperparameters, status, started_at, completed_at)
VALUES
-- Project 1: Sentiment
(
    'BERT-base Baseline',
    1, 1, 1, 'BERT',
    '{"lr":2e-5,"epochs":3,"batch_size":32,"max_len":128,"optimizer":"AdamW"}',
    'completed',
    '2026-03-01 09:00:00', '2026-03-01 11:30:00'
),
(
    'RoBERTa Fine-tuned',
    1, 1, 3, 'RoBERTa',
    '{"lr":1e-5,"epochs":5,"batch_size":16,"max_len":256,"optimizer":"AdamW","warmup_steps":500}',
    'completed',
    '2026-03-02 10:00:00', '2026-03-02 14:15:00'
),
(
    'DistilBERT Fast',
    1, 1, 1, 'DistilBERT',
    '{"lr":3e-5,"epochs":4,"batch_size":64,"max_len":128}',
    'completed',
    '2026-03-03 09:30:00', '2026-03-03 11:00:00'
),
-- Project 2: Medical CV
(
    'ResNet50 Transfer Learning',
    2, 2, 2, 'ResNet50',
    '{"lr":1e-4,"epochs":20,"batch_size":32,"pretrained":true,"dropout":0.3}',
    'completed',
    '2026-03-04 08:00:00', '2026-03-04 18:00:00'
),
(
    'DenseNet121 Medical',
    2, 2, 4, 'DenseNet121',
    '{"lr":5e-5,"epochs":25,"batch_size":16,"pretrained":true,"augmentation":true}',
    'completed',
    '2026-03-05 08:00:00', '2026-03-05 20:00:00'
),
-- Project 3: Forecasting
(
    'XGBoost Sales Baseline',
    3, 3, 5, 'XGBoost',
    '{"n_estimators":500,"max_depth":6,"learning_rate":0.05,"subsample":0.8}',
    'completed',
    '2026-03-06 10:00:00', '2026-03-06 11:00:00'
),
(
    'LSTM Time Series',
    3, 3, 5, 'LSTM',
    '{"units":128,"layers":2,"dropout":0.2,"epochs":50,"batch_size":64,"lookback":12}',
    'completed',
    '2026-03-07 09:00:00', '2026-03-07 13:00:00'
),
-- Project 5: Fraud
(
    'Random Forest Fraud Detector',
    5, 5, 6, 'RandomForest',
    '{"n_estimators":200,"max_depth":10,"class_weight":"balanced","min_samples_leaf":5}',
    'completed',
    '2026-03-08 10:00:00', '2026-03-08 10:45:00'
),
(
    'LightGBM Fraud v2',
    5, 5, 6, 'LightGBM',
    '{"n_estimators":500,"num_leaves":63,"learning_rate":0.05,"class_weight":"balanced"}',
    'running',
    '2026-03-11 09:00:00', NULL
),
-- Project 4: RAG
(
    'RAG with Chroma + GPT',
    4, 4, 3, 'RAG-Chroma',
    '{"embedding_model":"text-embedding-3-small","top_k":5,"chunk_size":512,"llm":"gpt-4o-mini"}',
    'completed',
    '2026-03-09 11:00:00', '2026-03-09 12:30:00'
);

-- ============================================
-- MODEL METRICS
-- ============================================
INSERT INTO model_metrics
    (experiment_id, split, accuracy, precision_score, recall, f1_score, loss, extra_metrics)
VALUES
-- BERT Baseline (exp 1)
(1, 'train',      0.9420, 0.9380, 0.9410, 0.9395, 0.1820, '{"auc":0.981}'),
(1, 'validation', 0.9110, 0.9050, 0.9080, 0.9065, 0.2540, '{"auc":0.962}'),
(1, 'test',       0.9050, 0.9010, 0.9030, 0.9020, 0.2610, '{"auc":0.959}'),

-- RoBERTa (exp 2)
(2, 'train',      0.9680, 0.9650, 0.9670, 0.9660, 0.1050, '{"auc":0.993}'),
(2, 'validation', 0.9430, 0.9400, 0.9420, 0.9410, 0.1720, '{"auc":0.978}'),
(2, 'test',       0.9380, 0.9350, 0.9370, 0.9360, 0.1810, '{"auc":0.975}'),

-- DistilBERT (exp 3)
(3, 'train',      0.9320, 0.9290, 0.9300, 0.9295, 0.2010, '{"auc":0.971}'),
(3, 'validation', 0.9050, 0.9010, 0.9030, 0.9020, 0.2680, '{"auc":0.952}'),
(3, 'test',       0.9010, 0.8980, 0.8990, 0.8985, 0.2720, '{"auc":0.949}'),

-- ResNet50 (exp 4)
(4, 'train',      0.8910, 0.8870, 0.8850, 0.8860, 0.3120, '{"auc_roc":0.941,"map":0.883}'),
(4, 'validation', 0.8640, 0.8600, 0.8580, 0.8590, 0.3870, '{"auc_roc":0.921}'),
(4, 'test',       0.8580, 0.8540, 0.8520, 0.8530, 0.3940, '{"auc_roc":0.917}'),

-- DenseNet121 (exp 5)
(5, 'train',      0.9150, 0.9120, 0.9100, 0.9110, 0.2640, '{"auc_roc":0.961}'),
(5, 'validation', 0.8890, 0.8850, 0.8830, 0.8840, 0.3290, '{"auc_roc":0.941}'),
(5, 'test',       0.8820, 0.8790, 0.8770, 0.8780, 0.3380, '{"auc_roc":0.937}'),

-- XGBoost (exp 6)
(6, 'train',      0.9520, 0.9480, 0.9440, 0.9460, 0.1340, '{"rmse":1820.50,"mape":0.084}'),
(6, 'validation', 0.9210, 0.9180, 0.9150, 0.9165, 0.2210, '{"rmse":2340.80,"mape":0.108}'),
(6, 'test',       0.9180, 0.9140, 0.9120, 0.9130, 0.2280, '{"rmse":2410.20,"mape":0.112}'),

-- LSTM (exp 7)
(7, 'train',      0.9380, 0.9340, 0.9310, 0.9325, 0.1720, '{"rmse":1680.40,"mape":0.078}'),
(7, 'validation', 0.9090, 0.9060, 0.9030, 0.9045, 0.2490, '{"rmse":2180.90,"mape":0.101}'),
(7, 'test',       0.9060, 0.9030, 0.9000, 0.9015, 0.2540, '{"rmse":2250.60,"mape":0.105}'),

-- RandomForest Fraud (exp 8)
(8, 'train',      0.9990, 0.9720, 0.9550, 0.9634, 0.0420, '{"auc_pr":0.883,"f1_fraud":0.872}'),
(8, 'validation', 0.9982, 0.9410, 0.9210, 0.9309, 0.0680, '{"auc_pr":0.841}'),
(8, 'test',       0.9980, 0.9380, 0.9180, 0.9279, 0.0720, '{"auc_pr":0.836}'),

-- RAG Pipeline (exp 10)
(10, 'test',      NULL, 0.8720, 0.8640, 0.8680, NULL,
    '{"hit_rate":0.91,"mrr":0.847,"faithfulness":0.923,"answer_relevancy":0.886}');

-- ============================================
-- VECTOR EMBEDDINGS (sample for RAG)
-- ============================================
INSERT INTO vector_embeddings (dataset_id, chunk_text, chunk_index, model_name, embedding_dim, metadata)
VALUES
(4, 'Q3 revenue increased by 12% YoY driven by SaaS subscriptions.',          1, 'text-embedding-3-small', 1536, '{"doc":"Q3_report.pdf","page":3}'),
(4, 'Customer churn rate dropped to 4.2% after loyalty program launch.',        2, 'text-embedding-3-small', 1536, '{"doc":"Q3_report.pdf","page":5}'),
(4, 'New partnerships with 3 Fortune 500 companies signed in October 2025.',    3, 'text-embedding-3-small', 1536, '{"doc":"strategy.pdf","page":1}'),
(4, 'Cloud infrastructure costs reduced by 18% via FinOps optimizations.',      4, 'text-embedding-3-small', 1536, '{"doc":"infra_review.pdf","page":2}'),
(4, 'AI-powered recommendation engine boosted CTR by 34% in A/B tests.',        5, 'text-embedding-3-small', 1536, '{"doc":"product_update.pdf","page":7}');

-- ============================================
-- VERIFICATION
-- ============================================
SELECT 'researchers'      AS table_name, COUNT(*) AS record_count FROM researchers
UNION ALL
SELECT 'ml_projects',   COUNT(*) FROM ml_projects
UNION ALL
SELECT 'datasets',      COUNT(*) FROM datasets
UNION ALL
SELECT 'experiments',   COUNT(*) FROM experiments
UNION ALL
SELECT 'model_metrics', COUNT(*) FROM model_metrics
UNION ALL
SELECT 'vector_embeddings', COUNT(*) FROM vector_embeddings;

DO $$
BEGIN
    RAISE NOTICE 'Sample data loaded successfully!';
    RAISE NOTICE '8 researchers | 6 projects | 6 datasets | 10 experiments | 19 metric rows | 5 embeddings';
END $$;
