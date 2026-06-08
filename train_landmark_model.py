import os
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

CSV_FILE = "asl_landmarks.csv"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "models/best_asl_model.pth"
LABELS_PATH = "models/labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

data = pd.read_csv(CSV_FILE)
X = data.iloc[:, 1:].values.astype(np.float32)
y = data.iloc[:, 0].values
print("\n===== CLASS DISTRIBUTION =====\n")

counts = Counter(y)
for cls, count in counts.items():
    print(cls, ":", count)

valid_classes = [cls for cls, count in counts.items() if count >= 10]

data = data[data.iloc[:, 0].isin(valid_classes)]

X = data.iloc[:, 1:].values.astype(np.float32)
y = data.iloc[:, 0].values

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

class_names = encoder.classes_

with open(LABELS_PATH, "w") as f:
    json.dump(class_names.tolist(), f)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ASLDataset(X_train, y_train)
val_dataset = ASLDataset(X_val, y_val)
test_dataset = ASLDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = ASLModel(len(class_names)).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

best_accuracy = 0

train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for val_inputs, val_labels in val_loader:

            val_inputs = val_inputs.to(DEVICE)
            val_labels = val_labels.to(DEVICE)

            val_outputs = model(val_inputs)

            _, val_preds = torch.max(val_outputs, 1)

            val_total += val_labels.size(0)

            val_correct += (val_preds == val_labels).sum().item()

    val_acc = val_correct / val_total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Acc: {epoch_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    scheduler.step(val_acc)

    epoch_path = f"models/asl_epoch_{epoch+1}.pth"

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": val_acc
    }, epoch_path)

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": val_acc
        }, MODEL_PATH)

        print("Best model saved!")

plt.figure(figsize=(8, 6))

plt.plot(train_losses)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig("results/loss_curve.png")

plt.close()

plt.figure(figsize=(8, 6))

plt.plot(train_accuracies, label="Train Accuracy")

plt.plot(val_accuracies, label="Validation Accuracy")

plt.title("Accuracy Curve")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("results/accuracy_curve.png")

plt.close()

checkpoint = torch.load(MODEL_PATH)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

SPLIT_CMAPS = {
    "train": "Purples",
    "validation": "Blues",
    "test": "Greens"
}

def plot_colored_confusion_matrix(cm, class_names, split_name):
    cmap = SPLIT_CMAPS.get(split_name, "Blues")

    plt.figure(figsize=(18, 15))

    sns.heatmap(
        cm,
        annot=False,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{split_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/{split_name}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved: results/{split_name}_confusion_matrix.png")

def print_per_class_accuracy(all_labels, all_preds, class_names, split_name):
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))

    print(f"\n{'='*55}")
    print(f"  Per-Letter Accuracy — {split_name}")
    print(f"{'='*55}")
    print(f"  {'Letter':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}  Bar")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}  ---")

    per_class_acc = []
    for i, cls in enumerate(class_names):
        total_cls = cm[i].sum()
        correct_cls = cm[i, i]
        acc = correct_cls / total_cls if total_cls > 0 else 0.0
        per_class_acc.append(acc)

        bar_len = int(acc * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {cls:<10} {correct_cls:>8} {total_cls:>8} {acc:>9.2%}  [{bar}]")

    mean_acc = np.mean(per_class_acc)
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'AVERAGE':<10} {'':>8} {'':>8} {mean_acc:>9.2%}")
    print(f"{'='*55}\n")

    with open(f"results/{split_name}_per_letter_accuracy.txt", "w") as f:
        f.write(f"Per-Letter Accuracy — {split_name}\n")
        f.write(f"{'='*45}\n")
        f.write(f"{'Letter':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}\n")
        f.write(f"{'-'*45}\n")
        for i, cls in enumerate(class_names):
            total_cls = cm[i].sum()
            correct_cls = cm[i, i]
            acc = correct_cls / total_cls if total_cls > 0 else 0.0
            f.write(f"{cls:<10} {correct_cls:>8} {total_cls:>8} {acc:>9.2%}\n")
        f.write(f"{'-'*45}\n")
        f.write(f"{'AVERAGE':<10} {'':>8} {'':>8} {mean_acc:>9.2%}\n")

    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.5), 5))
    colors = plt.cm.RdYlGn(np.array(per_class_acc))
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Letter", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Per-Letter Accuracy — {split_name}", fontsize=14)
    ax.axhline(y=mean_acc, color="blue", linestyle="--", linewidth=1.5, label=f"Average: {mean_acc:.2%}")
    ax.legend()

    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.0%}",
            ha="center", va="bottom", fontsize=7, rotation=45
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/{split_name}_per_letter_accuracy.png", dpi=150)
    plt.close()
    print(f"Per-letter accuracy chart saved: results/{split_name}_per_letter_accuracy.png")

num_classes = len(class_names)

def evaluate_model(loader, split_name):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\n{split_name} Accuracy: {accuracy:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    with open(f"results/{split_name}_classification_report.txt", "w") as f:

        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)

    plot_colored_confusion_matrix(cm, class_names, split_name)

    print_per_class_accuracy(all_labels, all_preds, class_names, split_name)

evaluate_model(train_loader, "train")

evaluate_model(val_loader, "validation")

evaluate_model(test_loader, "test")

print("\nTraining Finished Successfully!")

print("Best Validation Accuracy:", best_accuracy)