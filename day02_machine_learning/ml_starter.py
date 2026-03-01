"""
Day 02 — Machine Learning Starter
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To initiate a machine learning workflow using a real-world
dataset including preprocessing, model training, and evaluation.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "breast_cancer.csv")

df = pd.read_csv(file_path, header=None)

print("Dataset loaded successfully")

# Features & target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Day 02 ML workflow completed successfully.")
