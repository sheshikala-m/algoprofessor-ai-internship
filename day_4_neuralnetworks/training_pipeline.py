"""
Day 04 — Training Pipeline (MNIST Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To build a complete production-grade training pipeline for
deep neural networks on MNIST — covering data loading,
preprocessing, model training, evaluation, checkpointing
and result saving in one reusable pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report
import time


def load_data(base_dir, batch_size=64):
    print("--- 1. Loading MNIST Dataset (Auto Download) ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Batch size    : {batch_size}")
    return train_loader, test_loader


class PipelineModel(nn.Module):
    def __init__(self):
        super(PipelineModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class TrainingPipeline:
    def __init__(self, model, lr=0.001):
        self.model     = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5
        )
        self.history   = {"loss": [], "accuracy": [], "time": []}

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        correct    = 0
        total      = 0

        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss    = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == y_batch).sum().item()
            total      += y_batch.size(0)

        return epoch_loss / len(train_loader), correct / total

    def train(self, train_loader, epochs=10):
        print("\n--- 2. Running Training Pipeline ---")
        print(f"Epochs    : {epochs}")
        print(f"Optimizer : Adam")
        print(f"Scheduler : ReduceLROnPlateau\n")

        best_acc = 0
        for epoch in range(epochs):
            start     = time.time()
            loss, acc = self.train_epoch(train_loader)
            elapsed   = time.time() - start

            self.scheduler.step(loss)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)
            self.history["time"].append(elapsed)

            if acc > best_acc:
                best_acc = acc

            print(f"Epoch [{epoch+1:2d}/{epochs}] "
                  f"Loss: {loss:.4f} | Acc: {acc:.4f} | Time: {elapsed:.2f}s")

        print(f"\nBest Training Accuracy : {best_acc:.4f}")
        return self.history

    def evaluate(self, test_loader):
        print("\n--- 3. Evaluating Pipeline ---")
        self.model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs   = self.model(X_batch)
                predicted = outputs.argmax(dim=1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(y_batch.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy : {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        return all_preds, all_labels, accuracy

    def save_results(self, accuracy, output_dir):
        results = {
            "model"          : "PipelineModel",
            "dataset"        : "MNIST",
            "test_accuracy"  : round(accuracy, 4),
            "epochs"         : len(self.history["loss"]),
            "final_loss"     : round(self.history["loss"][-1], 4),
            "avg_epoch_time" : round(np.mean(self.history["time"]), 2)
        }
        save_path = os.path.join(output_dir, "results_pipeline.txt")
        with open(save_path, "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"Results saved to: {save_path}")


def visualize_results(history, base_dir):
    print("\n--- 4. Visualizing Pipeline Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Training Pipeline — MNIST Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(history["loss"], color="red", lw=2, marker="o")
    axes[0].set_title("Pipeline Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["accuracy"], color="blue", lw=2, marker="o")
    axes[1].set_title("Pipeline Training Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["time"], color="green", lw=2, marker="o")
    axes[2].set_title("Epoch Training Time (seconds)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)
    save_path = os.path.join(output_dir, "training_pipeline_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_training_pipeline():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    train_loader, test_loader = load_data(base_dir)
    model    = PipelineModel()
    pipeline = TrainingPipeline(model, lr=0.001)
    history  = pipeline.train(train_loader, epochs=10)
    all_preds, all_labels, accuracy = pipeline.evaluate(test_loader)
    pipeline.save_results(accuracy, output_dir)
    visualize_results(history, base_dir)

    print("\nDay 04 Training Pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()
