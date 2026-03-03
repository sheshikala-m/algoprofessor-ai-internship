# AlgoProfessor AI R&D Internship — Phase 1

Intern: Sheshikala Mamidisetti  
Batch: 2026  

---

# Breast Cancer Dataset Models

## 1. Linear Regression

### Objective
Built a Linear Regression model as a baseline to establish the preprocessing workflow and observe performance on a classification-type dataset.

### Model
Linear Regression (Scikit-learn)

### Results
- Accuracy: 81%  
- Used as a baseline before applying classification algorithms

### Dataset
Breast Cancer dataset (breast-cancer.csv)  
- 286 rows  
- 9 features  
- Target: recurrence / no recurrence  

### Tools Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 2. Logistic Regression

### Objective
Implemented Logistic Regression including preprocessing, model training, evaluation, and visualization.

### Model
Logistic Regression (Scikit-learn)

### Results
- Accuracy: 69%  
- ROC-AUC Score: 0.60  

### Dataset
Breast Cancer dataset (286 samples, 9 features)

### Output
Visualization saved in `outputs/logistic_regression_results.png`

### Tools Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

#  Heart Disease Prediction (Decision Tree)

## Objective
Built and optimized a Decision Tree classifier to predict heart disease using supervised machine learning.

## Dataset
Heart Disease dataset containing medical attributes such as age, cholesterol, blood pressure, etc.  
Target variable: presence or absence of heart disease.

## Workflow
1. Statistical data analysis  
2. Correlation heatmap  
3. Boxplot visualization  
4. Train-test split  
5. Decision Tree implementation  
6. Hyperparameter tuning (max_depth)  
7. Model evaluation  

## Model Performance
- Initial Accuracy: ~0.80  
- Best Accuracy: 0.836 (83.6%)  
- Best max_depth: 5  
- GridSearchCV and RandomizedSearchCV used for optimization  

## Observations
- Accuracy improved with moderate depth increase  
- Very deep trees increased complexity without performance gain  
- Risk of overfitting at higher depths  
- Simpler models provided stable and interpretable results  

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook  

## Conclusion
The tuned Decision Tree model achieved improved performance and demonstrated the importance of controlling model complexity to prevent overfitting.
