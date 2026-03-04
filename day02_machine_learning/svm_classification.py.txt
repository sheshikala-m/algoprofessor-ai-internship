"""
Day 02 — SVM Classification (Heart Disease Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To implement a Support Vector Machine classification model on a
real-world medical dataset including preprocessing, model training,
evaluation and visualization of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Load Dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "heart.csv")
df = pd.read_csv(file_path)
print("Dataset loaded successfully\n")

# Explore Data
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["target"].value_counts())

# Handle Missing Values
df.fillna(df.mode().iloc[0], inplace=True)
print("\nMissing values handled\n")

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    random_state=42
)
model.fit(X_train, y_train)
print("Model training completed\n")

# Prediction & Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions,
      target_names=["No Disease", "Heart Disease"]))

# Visualize AND Save
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "SVM Classification — Heart Disease\nSheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
    fontsize=13, fontweight="bold", y=1.02
)

# Chart 1 - Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            ax=axes[0, 0])
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

# Chart 3 - Prediction Distribution
axes[1, 0].hist(y_prob, bins=20, color="purple",
                edgecolor="white", alpha=0.8)
axes[1, 0].axvline(x=0.5, color="red", linestyle="--",
                   lw=2, label="Decision Boundary")
axes[1, 0].set_title("Prediction Probability Distribution",
                      fontweight="bold")
axes[1, 0].set_xlabel("Predicted Probability")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Chart 4 - Class Distribution
class_counts = y.value_counts()
axes[1, 1].bar(["No Disease", "Heart Disease"],
               class_counts.values,
               color=["green", "red"], width=0.5)
axes[1, 1].set_title("Class Distribution", fontweight="bold")
axes[1, 1].set_xlabel("Class")
axes[1, 1].set_ylabel("Number of Samples")
axes[1, 1].grid(True, alpha=0.3, axis="y")
for i, count in enumerate(class_counts.values):
    axes[1, 1].text(i, count + 1, str(count),
                    ha="center", fontweight="bold", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.3)

save_path = os.path.join(output_dir, "svm_results.png")
plt.savefig(save_path)
print(f"\nGraph saved to: {save_path}")
plt.show()
plt.close()

print("\nDay 02 SVM workflow completed successfully.")
