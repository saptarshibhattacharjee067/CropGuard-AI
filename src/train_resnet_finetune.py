import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights

from data.dataset import TomatoDataset


# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"

MODEL_PATH = "models/resnet18_finetuned_tomato.pth"

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 0.0001


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Dataset

train_dataset = TomatoDataset(
    MANIFEST,
    "train",
    use_augmentation=True
)

validation_dataset = TomatoDataset(
    MANIFEST,
    "validation",
    use_augmentation=False
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

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# Load pretrained ResNet18

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# Freeze pretrained layers

for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze final ResNet block

for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Replace final classification layer

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    10
)


# Move model to device

model = model.to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# Training

best_validation_accuracy = 0.0


for epoch in range(EPOCHS):

    # Training phase

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update trainable weights
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        correct / total * 100
    )

    # Validation phase

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

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += labels.size(0)


    validation_loss /= len(
        validation_loader
    )

    validation_accuracy = (
        validation_correct /
        validation_total *
        100
    )


    # Print results

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {validation_loss:.4f} "
        f"Val Acc: {validation_accuracy:.2f}%"
    )

    # Save best model

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best model saved! "
            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )

# Training complete

print("\nTraining complete!")

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print(
    f"Model saved to: {MODEL_PATH}"
)