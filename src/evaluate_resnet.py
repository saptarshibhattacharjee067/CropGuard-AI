import torch
from torch.utils.data import DataLoader

from data.dataset import TomatoDataset
from torchvision.models import resnet18
import torch.nn as nn

# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"
MODEL_PATH = "models/resnet18_tomato.pth"

BATCH_SIZE = 32

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

\
# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test dataset

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


# Load ResNet18

model = resnet18(
    weights=None
)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    10
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Evaluation

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


# Results

accuracy = correct / total * 100

print("\n" + "=" * 40)
print("ResNet18 TEST RESULTS")
print("=" * 40)

print("Test images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

print(
    f"Test Accuracy: {accuracy:.2f}%"
)