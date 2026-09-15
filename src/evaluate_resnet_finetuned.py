import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from data.dataset import TomatoDataset

# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"

MODEL_PATH = "models/resnet18_finetuned_tomato.pth"

BATCH_SIZE = 32

NUM_CLASSES = 10


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test Dataset

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


# Load ResNet18

model = resnet18(
    weights=None
)


# Replace final layer
number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    NUM_CLASSES
)


# Load our fine-tuned weights
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


# Final Result

accuracy = (
    correct / total
) * 100


print()
print("=" * 40)
print("Fine-Tuned ResNet18 TEST RESULTS")
print("=" * 40)

print("Test images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

print(
    f"Test Accuracy: {accuracy:.2f}%"
)