import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TomatoDataset
from models.cnn import TomatoCNN


# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Datasets

train_dataset = TomatoDataset(
    MANIFEST,
    "train"
)

validation_dataset = TomatoDataset(
    MANIFEST,
    "validation"
)


# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Model

model = TomatoCNN(
    num_classes=10
).to(device)


# Loss + Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training


for epoch in range(EPOCHS):


    # TRAINING

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    train_loss = (
        running_loss / len(train_loader)
    )


    # VALIDATION

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_accuracy = (
        100 * validation_correct
        / validation_total
    )

    validation_loss = (
        validation_loss
        / len(validation_loader)
    )


    # RESULTS

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {validation_loss:.4f} "
        f"Val Acc: {validation_accuracy:.2f}%"
    )


print("\nTraining complete!")

# Save trained model

MODEL_PATH = "models/tomato_cnn.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(f"Model saved to: {MODEL_PATH}")