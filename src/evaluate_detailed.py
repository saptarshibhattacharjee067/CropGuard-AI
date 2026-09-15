import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


from data.dataset import TomatoDataset
from models.cnn import TomatoCNN


# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"
MODEL_PATH = "models/tomato_cnn.pth"

BATCH_SIZE = 32


# Class names

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy"
]


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test dataset

test_dataset = TomatoDataset(
    MANIFEST,
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Load model

model = TomatoCNN(
    num_classes=10
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Collect predictions

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# Classification report

print("\nClassification Report")
print("=====================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# Confusion matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix")
print("================")

print(cm)


# Plot confusion matrix

plt.figure(
    figsize=(12, 10)
)

plt.imshow(cm)

plt.title(
    "Tomato Disease Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=90
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.colorbar()

plt.tight_layout()

plt.show()