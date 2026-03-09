"""
Day 04 — Transfer Learning (MNIST Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To implement Transfer Learning using a pretrained ResNet18
model fine-tuned on the MNIST dataset — demonstrating how
pretrained weights can be adapted to new tasks through
feature extraction and fine tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    print("--- 1. Loading MNIST Dataset (Auto Download) ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Resize to 32x32 and convert to 3 channels for ResNet18
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=False, download=True, transform=transform)

    # Use subset for faster training
    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    test_subset  = torch.utils.data.Subset(test_dataset,  range(2000))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_subset,  batch_size=64, shuffle=False)

    print(f"Train subset  : 10,000 samples")
    print(f"Test subset   : 2,000 samples")
    print(f"Image shape   : 3 x 32 x 32 (resized for ResNet18)")
    return train_loader, test_loader, base_dir


def build_transfer_model():
    print("\n--- 2. Building Transfer Learning Model (ResNet18) ---")
    print("Base model   : ResNet18")
    print("Strategy     : Freeze all layers, fine tune last block + FC")

    model = models.resnet18(pretrained=False)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block for fine tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC layer for 10 classes
    model.fc = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    return model


def train_model(model, train_loader, epochs=5):
    print("\n--- 3. Fine Tuning ResNet18 ---")
    print(f"Epochs    : {epochs}")
    print(f"Optimizer : Adam (lr=0.001)")
    print(f"Loss      : CrossEntropyLoss\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

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

        avg_loss = epoch_loss / len(train_loader)
        avg_acc  = correct / total
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    return model, train_losses, train_accs


def evaluate_model(model, test_loader):
    print("\n--- 4. Evaluating Transfer Learning Model ---")
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
    return all_preds, all_labels, accuracy


def visualize_results(train_losses, train_accs, all_labels, all_preds, base_dir):
    print("\n--- 5. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Transfer Learning (ResNet18) — MNIST Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(train_losses, color="red", lw=2, marker="o")
    axes[0].set_title("Transfer Learning Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, color="blue", lw=2, marker="o")
    axes[1].set_title("Transfer Learning Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[2])
    axes[2].set_title("Confusion Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)
    save_path = os.path.join(output_dir, "transfer_learning_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_transfer_learning():
    train_loader, test_loader, base_dir = load_data()
    model = build_transfer_model()
    model, train_losses, train_accs = train_model(model, train_loader, epochs=5)
    all_preds, all_labels, accuracy = evaluate_model(model, test_loader)
    visualize_results(train_losses, train_accs, all_labels, all_preds, base_dir)
    print("\nDay 04 Transfer Learning completed successfully.")


if __name__ == "__main__":
    run_transfer_learning()
