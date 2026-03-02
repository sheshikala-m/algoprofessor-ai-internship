"""
Day 02 — Machine Learning Starter (Breast Cancer Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To initiate a machine learning workflow using a real-world dataset,
including preprocessing, encoding, model training, and evaluation.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Dataset


base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "breast-cancer.csv")

df = pd.read_csv(file_path, header=None)

print("Dataset loaded successfully\n")


# Add Column Names


df.columns = [
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat",
    "class"
]


# Encode Categorical Data


encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

print("Categorical encoding completed\n")


# Features & Target


X = df.drop("class", axis=1)
y = df["class"]


# Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model Training


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Prediction & Evaluation


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nDay 02 ML workflow completed successfully.")
