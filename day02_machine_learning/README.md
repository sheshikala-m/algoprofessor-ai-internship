# AlgoProfessor AI R&D Internship — Phase 1

Intern: Sheshikala Mamidisetti  
Batch: 2026  

---

# Breast Cancer Dataset — Classification Models

## 1. Linear Regression (Baseline Model)

### Objective
Implemented Linear Regression as a baseline model to establish the preprocessing pipeline and evaluate how a regression-based approach performs on a classification-type dataset.

### Workflow
- Data loading and preprocessing  
- Handling categorical variables  
- Train-test split  
- Model training and evaluation  

### Results
- Accuracy: 81%  
- Served as an initial benchmark before applying classification algorithms  

### Dataset
Breast Cancer dataset (`breast-cancer.csv`)  
- 286 rows  
- 9 features  
- Target variable: recurrence / no recurrence  

### Key Insight
Linear Regression provided a baseline reference but is not ideal for classification problems.

---

## 2. Logistic Regression

### Objective
Applied Logistic Regression for binary classification including preprocessing, training, evaluation, and result visualization.

### Workflow
- Feature preprocessing  
- Train-test split  
- Model training  
- Evaluation using Accuracy and ROC-AUC  
- Visualization of results  

### Results
- Accuracy: 69%  
- ROC-AUC: 0.60  

### Output
`outputs/logistic_regression_results.png`

### Key Insight
Logistic Regression provided probabilistic outputs suitable for classification, though performance was moderate on this dataset.

---

# Heart Disease Dataset — Predictive Modeling

Dataset: `heart.csv`  
- 303 rows  
- 13 features  
- Target: presence or absence of heart disease  
- Source: UCI Machine Learning Repository  

---

## 1. Decision Tree Classifier

### Objective
Built and optimized a Decision Tree classifier to predict heart disease using supervised machine learning techniques.

### Workflow
- Exploratory Data Analysis (statistical summary, correlation heatmap, boxplots)  
- Train-test split  
- Model training  
- Hyperparameter tuning (max_depth)  
- Evaluation and visualization  

### Model Performance
- Initial Accuracy: ~0.80  
- Best Accuracy: 0.836  
- Optimal max_depth: 5  
- GridSearchCV and RandomizedSearchCV used for optimization  

### Observations
- Moderate depth improved accuracy  
- Very deep trees increased complexity without significant gain  
- Overfitting risk increases with high depth  
- Controlled complexity improves interpretability  

---

## 2. Random Forest Classifier

### Objective
Developed a Random Forest classifier to improve predictive performance using ensemble learning.

### Model Configuration
- n_estimators: 100  
- max_depth: 5  

### Evaluation Metrics

| Metric   | Score |
|----------|-------|
| Accuracy | 0.80  |
| ROC-AUC  | 0.91  |

### Output
`outputs/random_forest_results.png`

### Key Insight
Random Forest improved overall predictive stability and achieved a high ROC-AUC score, demonstrating the effectiveness of ensemble methods over single-tree models.

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---
Support Vector Machine (SVM) Classifier

Objective
Applied Support Vector Machine (SVM) for binary classification to identify an optimal decision boundary between classes.

Workflow
Data preprocessing and feature scaling
Train-test split
SVM model training
Evaluation using Accuracy and ROC-AUC

Results
Accuracy: ~0.82
ROC-AUC: ~0.88

Output
outputs/svm_classification_results.png

Key Insight
SVM showed strong performance on the dataset when proper feature scaling was applied, making it an effective classifier for structured healthcare data.

# Overall Conclusion

This phase focused on implementing and comparing multiple machine learning models across healthcare-related datasets. The experiments demonstrated:

- Importance of baseline modeling  
- Benefits of hyperparameter tuning  
- Impact of model complexity on performance  
- Strength of ensemble methods such as Random Forest  

The work reflects practical application of end-to-end machine learning workflows including preprocessing, modeling, evaluation, and interpretation.
