"""
Day 03 — Ensemble Models: XGBoost & LightGBM (Wine Quality Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To implement and compare XGBoost and LightGBM ensemble models on
the Wine Quality dataset — covering preprocessing, model training,
evaluation, feature importance analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nQuality distribution:")
    print(df["quality"].value_counts().sort_index())
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
    X_test = scaler.transform(X_test)

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test  set: {len(X_test)} samples")
    print("Preprocessing completed\n")
    return X_train, X_test, y_train, y_test, df


def train_xgboost(X_train, X_test, y_train, y_test):
    print("--- 3. Training XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    predictions = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"XGBoost Accuracy : {accuracy:.4f}")
    print(f"XGBoost ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions,
          target_names=["Bad Wine", "Good Wine"]))
    return xgb, predictions, y_prob, accuracy, roc_auc


def train_lightgbm(X_train, X_test, y_train, y_test):
    print("--- 4. Training LightGBM ---")
    lgbm = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    predictions = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"LightGBM Accuracy : {accuracy:.4f}")
    print(f"LightGBM ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions,
          target_names=["Bad Wine", "Good Wine"]))
    return lgbm, predictions, y_prob, accuracy, roc_auc


def print_comparison(xgb_acc, xgb_auc, lgbm_acc, lgbm_auc):
    print("\n" + "="*50)
    print("     XGBoost vs LightGBM COMPARISON")
    print("="*50)
    print(f"{'Model':<15} {'Accuracy':<12} {'ROC-AUC'}")
    print(f"{'-'*40}")
    print(f"{'XGBoost':<15} {xgb_acc:<12.4f} {xgb_auc:.4f}")
    print(f"{'LightGBM':<15} {lgbm_acc:<12.4f} {lgbm_auc:.4f}")
    print("="*50)
    if xgb_acc > lgbm_acc:
        print("Winner - XGBoost (higher accuracy)")
    elif lgbm_acc > xgb_acc:
        print("Winner - LightGBM (higher accuracy)")
    else:
        print("Tie - both models equal accuracy")
    print("="*50)


def visualize_results(df, xgb_model, lgbm_model,
                      xgb_pred, lgbm_pred,
                      xgb_prob, lgbm_prob,
                      y_test, base_dir):
    print("\n--- 5. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    feature_names = df.drop(["quality", "quality_label"], axis=1).columns.tolist()
    fpr_xgb,  tpr_xgb,  _ = roc_curve(y_test, xgb_prob)
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, lgbm_prob)
    xgb_auc  = roc_auc_score(y_test, xgb_prob)
    lgbm_auc = roc_auc_score(y_test, lgbm_prob)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "XGBoost vs LightGBM — Wine Quality Dataset\nSheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold", y=1.02
    )

    # Chart 1 - Confusion Matrix XGBoost
    cm_xgb = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Wine", "Good Wine"],
                yticklabels=["Bad Wine", "Good Wine"],
                ax=axes[0, 0])
    axes[0, 0].set_title("XGBoost — Confusion Matrix", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted Label")
    axes[0, 0].set_ylabel("Actual Label")

    # Chart 2 - Confusion Matrix LightGBM
    cm_lgbm = confusion_matrix(y_test, lgbm_pred)
    sns.heatmap(cm_lgbm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Bad Wine", "Good Wine"],
                yticklabels=["Bad Wine", "Good Wine"],
                ax=axes[0, 1])
    axes[0, 1].set_title("LightGBM — Confusion Matrix", fontweight="bold")
    axes[0, 1].set_xlabel("Predicted Label")
    axes[0, 1].set_ylabel("Actual Label")

    # Chart 3 - ROC Curves Both Models
    axes[1, 0].plot(fpr_xgb,  tpr_xgb,  color="blue",  lw=2, label=f"XGBoost  (AUC={xgb_auc:.3f})")
    axes[1, 0].plot(fpr_lgbm, tpr_lgbm, color="green", lw=2, label=f"LightGBM (AUC={lgbm_auc:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    axes[1, 0].set_title("ROC Curves — XGBoost vs LightGBM", fontweight="bold")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)

    # Chart 4 - Feature Importance Comparison (Normalized to same scale)
    xgb_importance  = xgb_model.feature_importances_
    lgbm_importance = lgbm_model.feature_importances_

    # Normalize both to 0-1 scale so both are visible
    xgb_norm  = xgb_importance  / xgb_importance.max()
    lgbm_norm = lgbm_importance / lgbm_importance.max()

    importance_df = pd.DataFrame({
        "Feature":  feature_names,
        "XGBoost":  xgb_norm,
        "LightGBM": lgbm_norm
    }).sort_values("XGBoost", ascending=True)

    x = np.arange(len(feature_names))
    width = 0.35
    axes[1, 1].barh(x - width/2, importance_df["XGBoost"],  width, color="blue",  label="XGBoost",  alpha=1.0)
    axes[1, 1].barh(x + width/2, importance_df["LightGBM"], width, color="green", label="LightGBM", alpha=0.7)
    axes[1, 1].set_yticks(x)
    axes[1, 1].set_yticklabels(importance_df["Feature"])
    axes[1, 1].set_title("Feature Importance Comparison (Normalized)", fontweight="bold")
    axes[1, 1].set_xlabel("Importance Score (Normalized 0-1)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    save_path = os.path.join(output_dir, "ensemble_models_results.png")
    plt.savefig(save_path)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_ensemble_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, base_dir = load_data()
    if df is None:
        return
    X_train, X_test, y_train, y_test, df = preprocess_data(df)
    xgb_model, xgb_pred, xgb_prob, xgb_acc, xgb_auc = train_xgboost(X_train, X_test, y_train, y_test)
    lgbm_model, lgbm_pred, lgbm_prob, lgbm_acc, lgbm_auc = train_lightgbm(X_train, X_test, y_train, y_test)
    print_comparison(xgb_acc, xgb_auc, lgbm_acc, lgbm_auc)
    visualize_results(df, xgb_model, lgbm_model,
                      xgb_pred, lgbm_pred,
                      xgb_prob, lgbm_prob,
                      y_test, base_dir)
    print("\nDay 03 Ensemble Models workflow completed successfully.")


if __name__ == "__main__":
    run_ensemble_models()
