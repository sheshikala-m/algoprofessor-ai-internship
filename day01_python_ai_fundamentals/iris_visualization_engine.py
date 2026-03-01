# Day 05 — Iris Visualization Automation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("iris_dataset.csv")

# Create outputs folder
output_path = "outputs"
os.makedirs(output_path, exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig(f"{output_path}/heatmap.png")
plt.close()

# 2. Boxplot
df.plot(kind="box", figsize=(10,6))
plt.title("Feature Distribution Boxplot")
plt.savefig(f"{output_path}/boxplot.png")
plt.close()

# 3. Pairplot
sns.pairplot(df, hue="target")
plt.savefig(f"{output_path}/pairplot.png")
plt.close()

print("All visualizations saved successfully.")
