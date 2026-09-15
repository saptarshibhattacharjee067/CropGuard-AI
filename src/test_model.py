import torch

from data.dataset import TomatoDataset
from torch.utils.data import DataLoader

from models.cnn import TomatoCNN


MANIFEST = "data/processed/tomato_manifest.csv"


# Dataset
train_dataset = TomatoDataset(
    MANIFEST,
    "train"
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


# Get one batch
images, labels = next(iter(train_loader))


# Create model
model = TomatoCNN(
    num_classes=10
)


# Run images through model
outputs = model(images)


print("Input shape:")
print(images.shape)

print("\nOutput shape:")
print(outputs.shape)

print("\nModel output:")
print(outputs)