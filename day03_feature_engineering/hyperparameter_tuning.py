"""
Day 03 — Hyperparameter Tuning: GridSearchCV & RandomizedSearchCV (Wine Quality Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To tune hyperparameters of Random Forest and Gradient Boosting models
using GridSearchCV and RandomizedSearchCV on the Wine Quality dataset —
covering best parameter search, evaluation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


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

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df, base_dir


def preprocess_data(df):
    print("\n--- 2. Preprocessing Data ---")
    df["quality_label"] = (df["quality"] >= 6).astype(int)
    print("Quality converted to binary:")
    print("  Good wine (quality >= 6) = 1")
    print("  Bad  wine (quality  < 6) = 0")
    print(df["quality_label"].value_counts())

    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test  set: {len(X_test)} samples")
    print("Preprocessing completed\n")
    return X_train, X_test, y_train, y_test


def tune_random_forest(X_train, X_test, y_train, y_test):
    print("--- 3. Tuning Random Forest with GridSearchCV ---")

    param_grid = {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [None, 5, 10],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_rf   = gs.best_estimator_
    preds     = best_rf.predict(X_test)
    y_prob    = best_rf.predict_proba(X_test)[:, 1]
    accuracy  = accuracy_score(y_test, preds)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"Best Params  : {gs.best_params_}")
    print(f"Best CV Score: {gs.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds,
          target_names=["Bad Wine", "Good Wine"]))

    return best_rf, accuracy, roc_auc, gs.best_params_, gs.best_score_, gs.cv_results_


def tune_gradient_boosting(X_train, X_test, y_train, y_test):
    print("--- 4. Tuning Gradient Boosting with RandomizedSearchCV ---")

    param_dist = {
        "n_estimators":  [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth":     [3, 4, 5, 6],
        "subsample":     [0.7, 0.8, 1.0]
    }

    gb = GradientBoostingClassifier(random_state=42)
    rs = RandomizedSearchCV(gb, param_dist, n_iter=20, cv=5,
                            scoring="accuracy", random_state=42,
                            n_jobs=-1, verbose=0)
    rs.fit(X_train, y_train)

    best_gb  = rs.best_estimator_
    preds    = best_gb.predict(X_test)
    y_prob   = best_gb.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, preds)
    roc_auc  = roc_auc_score(y_test, y_prob)

    print(f"Best Params  : {rs.best_params_}")
    print(f"Best CV Score: {rs.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds,
          target_names=["Bad Wine", "Good Wine"]))

    return best_gb, accuracy, roc_auc, rs.best_params_, rs.best_score_, rs.cv_results_


def print_comparison(rf_acc, rf_auc, rf_cv,
                     gb_acc, gb_auc, gb_cv):
    print("\n" + "="*55)
    print("     HYPERPARAMETER TUNING COMPARISON RESULTS")
    print("="*55)
    print(f"{'Model':<22} {'Accuracy':<12} {'ROC-AUC':<12} {'CV Score'}")
    print(f"{'-'*55}")
    print(f"{'Random Forest':<22} {rf_acc:<12.4f} {rf_auc:<12.4f} {rf_cv:.4f}")
    print(f"{'Gradient Boosting':<22} {gb_acc:<12.4f} {gb_auc:<12.4f} {gb_cv:.4f}")
    print("="*55)
    if rf_acc > gb_acc:
        print("Winner — Random Forest (higher accuracy after tuning)")
    elif gb_acc > rf_acc:
        print("Winner — Gradient Boosting (higher accuracy after tuning)")
    else:
        print("Tie — both models equal accuracy after tuning")
    print("="*55)


def visualize_results(rf_cv_results, gb_cv_results,
                      rf_acc, gb_acc,
                      rf_auc, gb_auc, base_dir):
    print("\n--- 5. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Hyperparameter Tuning — Wine Quality Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    # Chart 1 - GridSearchCV scores for Random Forest
    rf_means = rf_cv_results["mean_test_score"]
    rf_stds  = rf_cv_results["std_test_score"]
    axes[0, 0].errorbar(range(len(rf_means)), rf_means, yerr=rf_stds,
                        fmt="o-", color="#1565C0", ecolor="#90CAF9", capsize=4)
    axes[0, 0].axhline(max(rf_means), color="red", linestyle="--",
                       label=f"Best: {max(rf_means):.4f}")
    axes[0, 0].set_title("GridSearchCV — Random Forest", fontweight="bold")
    axes[0, 0].set_xlabel("Parameter Combination Index")
    axes[0, 0].set_ylabel("CV Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Chart 2 - RandomizedSearchCV scores for Gradient Boosting
    gb_means = gb_cv_results["mean_test_score"]
    gb_stds  = gb_cv_results["std_test_score"]
    axes[0, 1].errorbar(range(len(gb_means)), gb_means, yerr=gb_stds,
                        fmt="o-", color="#2E7D32", ecolor="#A5D6A7", capsize=4)
    axes[0, 1].axhline(max(gb_means), color="red", linestyle="--",
                       label=f"Best: {max(gb_means):.4f}")
    axes[0, 1].set_title("RandomizedSearchCV — Gradient Boosting", fontweight="bold")
    axes[0, 1].set_xlabel("Parameter Combination Index")
    axes[0, 1].set_ylabel("CV Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Chart 3 - Accuracy Comparison Bar Chart
    models   = ["Random Forest", "Gradient Boosting"]
    accuracy = [rf_acc, gb_acc]
    colors   = ["#1565C0", "#2E7D32"]
    bars = axes[1, 0].bar(models, accuracy, color=colors, width=0.4)
    axes[1, 0].set_title("Test Accuracy After Tuning", fontweight="bold")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0.5, 1.0)
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, accuracy):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center",
                        fontweight="bold", fontsize=11)

    # Chart 4 - ROC-AUC Comparison Bar Chart
    roc_scores = [rf_auc, gb_auc]
    bars2 = axes[1, 1].bar(models, roc_scores, color=colors, width=0.4)
    axes[1, 1].set_title("ROC-AUC After Tuning", fontweight="bold")
    axes[1, 1].set_ylabel("ROC-AUC Score")
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, roc_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center",
                        fontweight="bold", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=3.0)

    save_path = os.path.join(output_dir, "hyperparameter_tuning_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_hyperparameter_tuning():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, base_dir = load_data()
    if df is None:
        return

    X_train, X_test, y_train, y_test = preprocess_data(df)

    rf_model, rf_acc, rf_auc, rf_params, rf_cv, rf_cv_results = tune_random_forest(
        X_train, X_test, y_train, y_test)

    gb_model, gb_acc, gb_auc, gb_params, gb_cv, gb_cv_results = tune_gradient_boosting(
        X_train, X_test, y_train, y_test)

    print_comparison(rf_acc, rf_auc, rf_cv,
                     gb_acc, gb_auc, gb_cv)

    visualize_results(rf_cv_results, gb_cv_results,
                      rf_acc, gb_acc,
                      rf_auc, gb_auc, base_dir)

    print("\nDay 03 Hyperparameter Tuning workflow completed successfully.")


if __name__ == "__main__":
    run_hyperparameter_tuning()
