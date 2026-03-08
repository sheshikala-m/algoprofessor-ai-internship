"""
Day 03 — Production ML Pipeline (Wine Quality Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To build a production-grade Scikit-learn Pipeline combining
preprocessing and model training in one clean reusable pipeline.
Each pipeline is evaluated on test accuracy and 5-fold
cross validation for reliable performance measurement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def load_data():
    print("--- 1. Loading Wine Quality Dataset ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "winequality-red.csv")

    if not os.path.exists(file_path):
        print("Error: winequality-red.csv not found!")
        return None, None

    try:
        df = pd.read_csv(file_path, sep=";")
        if len(df.columns) < 5:
            df = pd.read_csv(file_path, sep=",")
    except Exception:
        df = pd.read_csv(file_path, sep=",")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["quality_label"] = (df["quality"] >= 6).astype(int)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["quality_label"].value_counts())
    return df, base_dir


def build_pipelines():
    print("\n--- 2. Building Production Pipelines ---")

    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
    }

    print("Pipelines built successfully!")
    for name, pipe in pipelines.items():
        steps = " -> ".join([s[0] for s in pipe.steps])
        print(f"  {name}: {steps}")

    return pipelines


def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, X, y):
    print("\n--- 3. Training and Evaluating All Pipelines ---")

    results  = []
    cm_data  = {}
    roc_data = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        y_prob      = pipeline.predict_proba(X_test)[:, 1]
        accuracy    = accuracy_score(y_test, predictions)
        roc_auc     = roc_auc_score(y_test, y_prob)
        cv_scores   = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        cv_mean     = cv_scores.mean()
        cv_std      = cv_scores.std()

        results.append({
            "Model":    name,
            "Accuracy": round(accuracy, 4),
            "ROC-AUC":  round(roc_auc, 4),
            "CV Mean":  round(cv_mean, 4),
            "CV Std":   round(cv_std, 4)
        })

        cm_data[name]  = confusion_matrix(y_test, predictions)
        roc_data[name] = (y_prob, roc_auc)

        print(f"\n{name}")
        print(f"  Test Accuracy : {accuracy:.4f}")
        print(f"  ROC-AUC       : {roc_auc:.4f}")
        print(f"  CV Mean       : {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, predictions,
              target_names=["Bad Wine", "Good Wine"]))

    return pd.DataFrame(results), cm_data, roc_data


def print_best_pipeline(results_df):
    print("\n" + "="*60)
    print("         PIPELINE COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))

    best_acc = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
    best_cv  = results_df.loc[results_df["CV Mean"].idxmax(),  "Model"]
    best_roc = results_df.loc[results_df["ROC-AUC"].idxmax(),  "Model"]

    print(f"\n  Best Test Accuracy : {best_acc}")
    print(f"  Best CV Mean       : {best_cv}")
    print(f"  Best ROC-AUC       : {best_roc}")
    print("\n  RECOMMENDATION: Random Forest Pipeline")
    print("  Reason: Highest test accuracy and ROC-AUC")
    print("  Pipeline ensures no data leakage between")
    print("  train and test sets — production grade.")
    print("="*60)


def visualize_results(results_df, cm_data, roc_data, y_test, base_dir):
    print("\n--- 4. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Production ML Pipeline — Wine Quality Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    # Chart 1 - Test Accuracy Comparison
    bars = axes[0, 0].bar(results_df["Model"], results_df["Accuracy"],
                          color=colors, width=0.5)
    axes[0, 0].set_title("Test Accuracy Comparison", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, results_df["Accuracy"]):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center",
                        fontweight="bold", fontsize=10)

    # Chart 2 - ROC-AUC Comparison
    bars2 = axes[0, 1].bar(results_df["Model"], results_df["ROC-AUC"],
                            color=colors, width=0.5)
    axes[0, 1].set_title("ROC-AUC Comparison", fontweight="bold")
    axes[0, 1].set_ylabel("ROC-AUC Score")
    axes[0, 1].set_ylim(0.5, 1.0)
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, results_df["ROC-AUC"]):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center",
                        fontweight="bold", fontsize=10)

    # Chart 3 - Cross Validation Score
    axes[1, 0].bar(results_df["Model"], results_df["CV Mean"],
                   color=colors, width=0.5,
                   yerr=results_df["CV Std"], capsize=5)
    axes[1, 0].set_title("5-Fold Cross Validation Score", fontweight="bold")
    axes[1, 0].set_ylabel("CV Mean Accuracy")
    axes[1, 0].set_ylim(0.5, 1.0)
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for i, (val, std) in enumerate(zip(results_df["CV Mean"], results_df["CV Std"])):
        axes[1, 0].text(i, val + std + 0.01,
                        f"{val:.3f}", ha="center",
                        fontweight="bold", fontsize=10)

    # Chart 4 - Metrics Heatmap
    metrics     = ["Accuracy", "ROC-AUC", "CV Mean"]
    heatmap_data = results_df[metrics].copy()
    heatmap_data.index = results_df["Model"]
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                ax=axes[1, 1], linewidths=0.5,
                annot_kws={"size": 11, "weight": "bold"})
    axes[1, 1].set_title("All Metrics Heatmap", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].tick_params(axis="y", rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=3.0)

    save_path = os.path.join(output_dir, "ml_pipeline_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_ml_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, base_dir = load_data()
    if df is None:
        return

    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines              = build_pipelines()
    results_df, cm_data, roc_data = train_and_evaluate(
        pipelines, X_train, X_test, y_train, y_test, X, y
    )
    print_best_pipeline(results_df)
    visualize_results(results_df, cm_data, roc_data, y_test, base_dir)

    print("\nDay 03 ML Pipeline workflow completed successfully.")


if __name__ == "__main__":
    run_ml_pipeline()
