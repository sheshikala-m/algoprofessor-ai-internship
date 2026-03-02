"""
Day 02 —  logistic_regression(Breast Cancer Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To implement Logistic Regression on a real-world medical dataset,
including preprocessing, encoding, model training, evaluation,
and visualization of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Load Dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "breast-cancer.csv")
df = pd.read_csv(file_path, header=None)
print("Dataset loaded successfully\n")

# Add Column Names
df.columns = [
    "age", "menopause", "tumor_size", "inv_nodes",
    "node_caps", "deg_malig", "breast",
    "breast_quad", "irradiat", "class"
]

# Explore Data
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["class"].value_counts())

# Handle Missing Values (replace '?' with mode)
df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
print("\nMissing values handled\n")

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model training completed\n")

# Prediction & Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions,
                                target_names=["No Recurrence", "Recurrence"])

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(report)

# Visualize AND Save
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Logistic Regression — Breast Cancer Classification\nSheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
    fontsize=13, fontweight="bold", y=1.02
)

# Chart 1 - Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Recurrence", "Recurrence"],
            yticklabels=["No Recurrence", "Recurrence"],
            ax=axes[0, 0], linewidths=0.5)
axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("Actual Label")

# Chart 2 - ROC Curve
axes[0, 1].plot(fpr, tpr, color="blue", lw=2,
                label=f"ROC Curve (AUC = {roc_auc:.3f})")
axes[0, 1].plot([0, 1], [0, 1], color="gray", linestyle="--",
                label="Random Classifier")
axes[0, 1].set_title("ROC-AUC Curve", fontweight="bold")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# Chart 3 - Feature Importance
feature_names = X.columns.tolist()
coefficients = model.coef_[0]
sorted_idx = np.argsort(np.abs(coefficients))[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_coefs = coefficients[sorted_idx]
colors = ["blue" if c > 0 else "red" for c in sorted_coefs]

axes[1, 0].barh(sorted_features, sorted_coefs, color=colors)
axes[1, 0].axvline(x=0, color="black", linewidth=0.8, linestyle="--")
axes[1, 0].set_title("Feature Importance (Coefficients)", fontweight="bold")
axes[1, 0].set_xlabel("Coefficient Value")
axes[1, 0].grid(True, alpha=0.3, axis="x")

# Chart 4 - Class Distribution
class_counts = y.value_counts()
class_labels = ["No Recurrence", "Recurrence"]
axes[1, 1].bar(class_labels, class_counts.values,
               color=["green", "red"], width=0.5)
axes[1, 1].set_title("Class Distribution", fontweight="bold")
axes[1, 1].set_xlabel("Class")
axes[1, 1].set_ylabel("Number of Samples")
axes[1, 1].grid(True, alpha=0.3, axis="y")

for i, count in enumerate(class_counts.values):
    axes[1, 1].text(i, count + 1, str(count), ha="center",
                    fontweight="bold", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.97])

save_path = f"{output_dir}/logistic_regression_results.png"
plt.savefig(save_path)
print(f"\nGraph saved to: {save_path}")
plt.show()

print("\nDay 02 ML workflow completed successfully.")
