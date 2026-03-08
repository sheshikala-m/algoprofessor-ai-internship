"""
Day 04 — Neural Network from Scratch (MNIST Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To build a Deep Neural Network completely from scratch using
PyTorch with statistical theory grounding — implementing
forward pass, backward propagation, loss calculation,
descriptive statistics and linear algebra operations manually.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    print("--- 1. Loading MNIST Dataset (Auto Download) ---")
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Classes       : {train_dataset.classes}")
    return train_loader, test_loader, base_dir


def statistical_analysis(train_loader):
    print("\n--- 2. Statistical Theory Grounding ---")
    all_data = []
    for X_batch, _ in train_loader:
        all_data.append(X_batch.numpy().flatten())
        if len(all_data) > 10:
            break
    pixel_data = np.concatenate(all_data)
    print(f"Mean pixel value : {np.mean(pixel_data):.4f}")
    print(f"Std pixel value  : {np.std(pixel_data):.4f}")
    print(f"Min pixel value  : {np.min(pixel_data):.4f}")
    print(f"Max pixel value  : {np.max(pixel_data):.4f}")
    print(f"Variance         : {np.var(pixel_data):.4f}")


def linear_algebra_operations():
    print("\n--- 3. Linear Algebra Operations ---")
    W = np.random.randn(784, 512)
    x = np.random.randn(10, 784)
    result = x @ W
    print(f"Input matrix shape  : {x.shape}")
    print(f"Weight matrix shape : {W.shape}")
    print(f"Output shape        : {result.shape}")
    print(f"Matrix rank         : {np.linalg.matrix_rank(W)}")
    print("Linear algebra operations completed")


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(DeepNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def train_model(train_loader, epochs=10):
    print("\n--- 4. Training Deep Neural Network ---")
    print(f"Architecture : 784 -> 512 -> 256 -> 128 -> 64 -> 10")
    print(f"Activation   : ReLU + BatchNorm + Dropout")
    print(f"Loss         : CrossEntropyLoss")
    print(f"Optimizer    : Adam (lr=0.001)")
    print(f"Epochs       : {epochs}\n")

    model     = DeepNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses = []
    train_accs   = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct    = 0
        total      = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == y_batch).sum().item()
            total      += y_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        avg_acc  = correct / total
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        print(f"Epoch [{epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    return model, train_losses, train_accs


def evaluate_model(model, test_loader):
    print("\n--- 5. Evaluating Model ---")
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs   = model(X_batch)
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy : {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    return all_preds, all_labels, accuracy


def visualize_results(train_losses, train_accs, all_labels, all_preds, base_dir):
    print("\n--- 6. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Deep Neural Network from Scratch — MNIST Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(train_losses, color="red", lw=2, marker="o")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, color="blue", lw=2, marker="o")
    axes[1].set_title("Training Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2])
    axes[2].set_title("Confusion Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)
    save_path = os.path.join(output_dir, "neural_network_scratch_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_neural_network_scratch():
    train_loader, test_loader, base_dir = load_data()
    statistical_analysis(train_loader)
    linear_algebra_operations()
    model, train_losses, train_accs = train_model(train_loader, epochs=10)
    all_preds, all_labels, accuracy = evaluate_model(model, test_loader)
    visualize_results(train_losses, train_accs, all_labels, all_preds, base_dir)
    print("\nDay 04 Neural Network from Scratch completed successfully.")


if __name__ == "__main__":
    run_neural_network_scratch()
