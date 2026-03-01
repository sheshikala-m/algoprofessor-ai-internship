"""
Day 05 — Iris Visualization Automation
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To automate exploratory data visualization by generating
correlation heatmaps, boxplots, and pairplots for the
Iris dataset, enabling efficient visual analysis and
structured insight generation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset safely
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "iris_dataset.csv")
df = pd.read_csv(file_path)

# Create outputs folder
output_path = os.path.join(base_dir, "outputs")
os.makedirs(output_path, exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
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

