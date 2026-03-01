"""
Day 06 — Iris OOP Analysis Engine
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To design a modular Object-Oriented data analysis engine
for scalable dataset inspection, statistical evaluation,
and visualization automation.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Fix display cutoff in Colab/Jupyter
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class IrisAnalysis:
    """
    A class-based analysis engine for performing structured
    exploratory data analysis on the Iris dataset.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

# Load Dataset

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Dataset loaded successfully\n")


# Statistical Summary

    def show_summary(self):
        print("Statistical Summary:\n")
        return self.df.describe()


 # Correlation Matrix

    def show_correlation(self):
        print("\nCorrelation Matrix:\n")
        return self.df.corr(numeric_only=True)

# Heatmap Visualization

    def create_heatmap(self):

# Create outputs folder if not exists
        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(8,6))

        sns.heatmap(
            self.df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title("Iris Correlation Heatmap")
        plt.tight_layout()

        plt.savefig("outputs/oop_heatmap.png")
        plt.show()

        print("\nHeatmap saved in outputs folder")



# Main Execution

analysis = IrisAnalysis("iris_dataset.csv")

analysis.load_data()

summary = analysis.show_summary()
summary

correlation = analysis.show_correlation()
correlation

analysis.create_heatmap()

print("\nOOP Analysis completed successfully")

if __name__ == "__main__":
    main()
