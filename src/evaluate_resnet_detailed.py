import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models import resnet18

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

from data.dataset import TomatoDataset


# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"
MODEL_PATH = "models/resnet18_tomato.pth"

BATCH_SIZE = 32


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Dataset

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


# Class names

class_names = [
    test_dataset.idx_to_class[i]
    for i in range(len(test_dataset.idx_to_class))
]


# Create ResNet18

model = resnet18(
    weights=None
)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    10
)


# Load our trained weights

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Collect predictions

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# Classification report

print("\n")
print("ResNet18 Classification Report")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
)


# Confusion matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix")
print("=" * 60)

print(cm)

# Plot confusion matrix

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.title("ResNet18 Tomato Disease Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/resnet18_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved to:")
print("results/resnet18_confusion_matrix.png")