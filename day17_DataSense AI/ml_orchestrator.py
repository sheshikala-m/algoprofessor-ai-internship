"""
ml_orchestrator.py — MLOrchestrator Agent
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    silhouette_score,
    accuracy_score
)


class MLOrchestratorAgent:

    def __init__(self, output_dir="outputs"):
        self.name = "MLOrchestrator"
        self.output_dir = output_dir
        self.df = None
        self.ml_results = {}
        self.scaler = StandardScaler()

        os.makedirs(output_dir, exist_ok=True)

        print(f"[{self.name}] Agent initialized")

    # =========================================================
    # RECEIVE DATA
    # =========================================================
    def receive_data(self, agent_message):

        self.df = agent_message["df_ref"]
        self.numeric_cols = agent_message["numeric_cols"]
        self.cat_cols = agent_message["cat_cols"]
        self.stats_report = agent_message.get("stats_report", {})

        print(f"[{self.name}] Received data from {agent_message['from_agent']}")

    # =========================================================
    # PREPROCESS FEATURES
    # =========================================================
    def preprocess_features(self, target_col="high_value"):

        print(f"[{self.name}] Preprocessing features...")

        feature_cols = [c for c in self.numeric_cols if c != target_col]

        X = self.df[feature_cols].fillna(0)

        X_scaled = self.scaler.fit_transform(X)

        self.feature_cols = feature_cols
        self.X = X
        self.X_scaled = X_scaled

        if target_col in self.df.columns:
            self.y = self.df[target_col]
        else:
            self.y = None

        print(f"[{self.name}] Feature matrix shape: {X_scaled.shape}")

        return X_scaled, self.y

    # =========================================================
    # PCA
    # =========================================================
    def run_pca(self, n_components=3):

        print(f"[{self.name}] Running PCA...")

        pca = PCA(n_components=n_components, random_state=42)

        pca_result = pca.fit_transform(self.X_scaled)

        explained = pca.explained_variance_ratio_

        result = {
            "n_components": n_components,
            "explained_variance_ratio": [
                round(float(v), 4) for v in explained
            ],
            "cumulative_variance": round(float(sum(explained)), 4),
        }

        self.pca_result = pca_result

        self.ml_results["pca"] = result

        print(
            f"[{self.name}] Cumulative variance explained: "
            f"{result['cumulative_variance']:.1%}"
        )

        self._plot_pca(pca_result, explained)

        return result

    def _plot_pca(self, pca_result, explained):

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            alpha=0.5,
            c="#4A90D9",
            s=15
        )

        axes[0].set_xlabel(f"PC1 ({explained[0]:.1%} var)")
        axes[0].set_ylabel(f"PC2 ({explained[1]:.1%} var)")
        axes[0].set_title("PCA: PC1 vs PC2")

        components = list(range(1, len(explained) + 1))

        axes[1].bar(
            components,
            [e * 100 for e in explained]
        )

        axes[1].plot(
            components,
            [e * 100 for e in explained],
            "o-"
        )

        axes[1].set_xlabel("Principal Component")
        axes[1].set_ylabel("Variance Explained (%)")
        axes[1].set_title("Scree Plot")

        plt.tight_layout()

        path = os.path.join(self.output_dir, "pca_analysis.png")

        plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close()

        print(f"[{self.name}] PCA chart saved: {path}")

    # =========================================================
    # KMEANS
    # =========================================================
    def run_kmeans(self, n_clusters=4):

        print(f"[{self.name}] Running KMeans clustering...")

        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(self.X_scaled)

        sil_score = silhouette_score(self.X_scaled, labels)

        cluster_counts = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        result = {
            "n_clusters": n_clusters,
            "silhouette_score": round(float(sil_score), 4),
            "cluster_sizes": {
                str(k): int(v)
                for k, v in cluster_counts.items()
            },
            "inertia": round(float(km.inertia_), 2),
        }

        self.cluster_labels = labels

        self.ml_results["kmeans"] = result

        print(f"[{self.name}] Silhouette Score: {sil_score:.4f}")

        self._plot_clusters(labels, n_clusters)

        return result

    def _plot_clusters(self, labels, n_clusters):

        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        fig, ax = plt.subplots(figsize=(10, 6))

        for k in range(n_clusters):

            mask = labels == k

            ax.scatter(
                self.pca_result[mask, 0],
                self.pca_result[mask, 1],
                s=20,
                alpha=0.6,
                color=colors[k],
                label=f"Cluster {k}"
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("KMeans Clusters")

        ax.legend()

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "kmeans_clusters.png"
        )

        plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close()

        print(f"[{self.name}] Cluster chart saved: {path}")

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    def run_random_forest(self):

        if self.y is None:
            print(f"[{self.name}] No target column found")
            return {}

        print(f"[{self.name}] Training Random Forest...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(
            rf,
            self.X_scaled,
            self.y,
            cv=5
        )

        importance = dict(
            zip(self.feature_cols, rf.feature_importances_)
        )

        top_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:8]

        result = {
            "accuracy": round(float(acc), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "top_features": [
                {
                    "feature": k,
                    "importance": round(float(v), 4)
                }
                for k, v in top_features
            ]
        }

        self.ml_results["random_forest"] = result

        print(f"[{self.name}] Accuracy: {acc:.4f}")

        self._plot_feature_importance(top_features)

        return result

    def _plot_feature_importance(self, top_features):

        features, importances = zip(*top_features)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            list(reversed(features)),
            list(reversed(importances))
        )

        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")

        plt.tight_layout()

        path = os.path.join(
            self.output_dir,
            "feature_importance.png"
        )

        plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close()

        print(f"[{self.name}] Feature importance chart saved: {path}")

    # =========================================================
    # SVM
    # =========================================================
    def run_svm(self):

        if self.y is None:
            return {}

        print(f"[{self.name}] Training SVM...")

        sample_idx = np.random.choice(
            len(self.X_scaled),
            min(1000, len(self.X_scaled)),
            replace=False
        )

        Xs = self.X_scaled[sample_idx]

        ys = self.y.iloc[sample_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            Xs,
            ys,
            test_size=0.2,
            random_state=42
        )

        svm = SVC(
            kernel="rbf",
            C=1.0,
            random_state=42
        )

        svm.fit(X_train, y_train)

        acc = accuracy_score(
            y_test,
            svm.predict(X_test)
        )

        result = {
            "accuracy": round(float(acc), 4),
            "kernel": "rbf",
            "C": 1.0
        }

        self.ml_results["svm"] = result

        print(f"[{self.name}] SVM Accuracy: {acc:.4f}")

        return result

    # =========================================================
    # EXPORT JSON
    # =========================================================
    def export_ml_json(self):

        path = os.path.join(
            self.output_dir,
            "ml_results.json"
        )

        with open(path, "w") as f:
            json.dump(
                self.ml_results,
                f,
                indent=2,
                default=str
            )

        print(f"[{self.name}] ML results saved: {path}")

        return path

    # =========================================================
    # AGENT MESSAGE
    # =========================================================
    def get_agent_message(self):

        return {
            "from_agent": self.name,
            "status": "complete",
            "ml_results": self.ml_results,
            "df_ref": self.df,
            "numeric_cols": self.numeric_cols,
            "cat_cols": self.cat_cols,
            "stats_report": self.stats_report,
        }