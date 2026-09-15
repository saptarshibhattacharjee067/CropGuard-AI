import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models import resnet18

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from data.dataset import TomatoDataset


# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"

MODEL_PATH = "models/resnet18_finetuned_tomato.pth"

OUTPUT_PATH = "results/resnet18_finetuned_confusion_matrix.png"

BATCH_SIZE = 32

NUM_CLASSES = 10


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================
# Load test dataset
# ============================================

test_dataset = TomatoDataset(
    MANIFEST,
    "test",
    use_augmentation=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# Load fine-tuned ResNet18

model = resnet18(
    weights=None
)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

# Store predictions

all_labels = []

all_predictions = []


# Run predictions

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# Class names

class_names = [
    test_dataset.idx_to_class[i]
    for i in range(NUM_CLASSES)
]


# Classification Report

print()
print("=" * 70)
print("Fine-Tuned ResNet18 Classification Report")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4
)

print(report)


# Confusion Matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("=" * 70)
print("Confusion Matrix")
print("=" * 70)

print(cm)


# Calculate accuracy

correct = sum(
    true == predicted
    for true, predicted
    in zip(
        all_labels,
        all_predictions
    )
)

total = len(all_labels)

accuracy = (
    correct / total
) * 100


print()
print("=" * 70)
print("Final Test Accuracy")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Incorrect predictions: {total - correct}"
)


# Plot Confusion Matrix

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Fine-Tuned ResNet18 Tomato Disease Confusion Matrix"
)

plt.colorbar()

tick_marks = range(NUM_CLASSES)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# Add numbers inside matrix
for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


# Save confusion matrix

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print()
print(
    f"Confusion matrix saved to: {OUTPUT_PATH}"
)