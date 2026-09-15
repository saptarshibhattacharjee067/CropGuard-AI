import torch
from torch.utils.data import DataLoader

from data.dataset import TomatoDataset
from models.cnn import TomatoCNN

# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"
MODEL_PATH = "models/tomato_cnn.pth"

BATCH_SIZE = 32

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


# Evaluate

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total


# Result

print("\nTest Results")
print("--------------------")
print("Test images:", total)
print(f"Test Accuracy: {accuracy:.2f}%")