from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from data.dataset import TomatoDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tomato_manifest.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_finetuned_tomato.pth"
)

TEMPERATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "calibration_temperature.json"
)

BATCH_SIZE = 32

NUMBER_OF_CLASSES = 10

NUMBER_OF_BINS = 10


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


with open(
    TEMPERATURE_PATH,
    "r",
    encoding="utf-8"
) as file:

    calibration_data = json.load(
        file
    )


temperature = calibration_data[
    "temperature"
]


print(
    f"Loaded temperature: "
    f"{temperature:.4f}"
)


test_dataset = TomatoDataset(
    str(MANIFEST),
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = resnet18(
    weights=None
)

number_of_features = (
    model.fc.in_features
)

model.fc = nn.Linear(
    number_of_features,
    NUMBER_OF_CLASSES
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model = model.to(device)

model.eval()


all_logits = []

all_labels = []


print()
print(
    "Collecting test predictions..."
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )

        all_logits.append(
            outputs.cpu()
        )

        all_labels.append(
            labels.cpu()
        )


logits = torch.cat(
    all_logits
)

labels = torch.cat(
    all_labels
)


criterion = nn.CrossEntropyLoss()


uncalibrated_loss = criterion(
    logits,
    labels
).item()


uncalibrated_predictions = (
    logits.argmax(
        dim=1
    )
)


uncalibrated_accuracy = (
    (
        uncalibrated_predictions
        == labels
    )
    .float()
    .mean()
    .item()
    * 100
)


calibrated_logits = (
    logits
    / temperature
)


calibrated_loss = criterion(
    calibrated_logits,
    labels
).item()


calibrated_predictions = (
    calibrated_logits.argmax(
        dim=1
    )
)


calibrated_accuracy = (
    (
        calibrated_predictions
        == labels
    )
    .float()
    .mean()
    .item()
    * 100
)


def calculate_ece(
    logits,
    labels,
    number_of_bins=10
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidences, predictions = (
        probabilities.max(
            dim=1
        )
    )

    correct = (
        predictions == labels
    ).float()


    bin_boundaries = torch.linspace(
        0,
        1,
        number_of_bins + 1
    )


    ece = 0.0


    for i in range(
        number_of_bins
    ):

        lower = (
            bin_boundaries[i]
        )

        upper = (
            bin_boundaries[i + 1]
        )


        if i == number_of_bins - 1:

            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )

        else:

            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )


        count = mask.sum().item()


        if count == 0:

            continue


        bin_confidence = (
            confidences[mask]
            .mean()
            .item()
        )

        bin_accuracy = (
            correct[mask]
            .mean()
            .item()
        )


        ece += (
            count
            / len(labels)
        ) * abs(
            bin_accuracy
            - bin_confidence
        )


    return ece


uncalibrated_ece = calculate_ece(
    logits,
    labels,
    NUMBER_OF_BINS
)


calibrated_ece = calculate_ece(
    calibrated_logits,
    labels,
    NUMBER_OF_BINS
)


confidence_uncalibrated = (
    torch.softmax(
        logits,
        dim=1
    )
    .max(
        dim=1
    )
    .values
    * 100
)


confidence_calibrated = (
    torch.softmax(
        calibrated_logits,
        dim=1
    )
    .max(
        dim=1
    )
    .values
    * 100
)


correct_mask = (
    uncalibrated_predictions
    == labels
)

incorrect_mask = (
    ~correct_mask
)


average_correct_confidence_before = (
    confidence_uncalibrated[
        correct_mask
    ]
    .mean()
    .item()
)


average_incorrect_confidence_before = (
    confidence_uncalibrated[
        incorrect_mask
    ]
    .mean()
    .item()
)


average_correct_confidence_after = (
    confidence_calibrated[
        correct_mask
    ]
    .mean()
    .item()
)


average_incorrect_confidence_after = (
    confidence_calibrated[
        incorrect_mask
    ]
    .mean()
    .item()
)


maximum_incorrect_confidence_before = (
    confidence_uncalibrated[
        incorrect_mask
    ]
    .max()
    .item()
)


maximum_incorrect_confidence_after = (
    confidence_calibrated[
        incorrect_mask
    ]
    .max()
    .item()
)


print()
print("=" * 70)
print("CROPGUARD AI FINAL CALIBRATED TEST EVALUATION")
print("=" * 70)


print()
print(
    f"Test images: {len(labels)}"
)


print()
print("BEFORE CALIBRATION")
print("-" * 70)

print(
    f"Accuracy: "
    f"{uncalibrated_accuracy:.2f}%"
)

print(
    f"NLL: "
    f"{uncalibrated_loss:.4f}"
)

print(
    f"ECE: "
    f"{uncalibrated_ece:.4f}"
)

print(
    f"Average confidence on correct predictions: "
    f"{average_correct_confidence_before:.2f}%"
)

print(
    f"Average confidence on incorrect predictions: "
    f"{average_incorrect_confidence_before:.2f}%"
)

print(
    f"Maximum confidence on incorrect predictions: "
    f"{maximum_incorrect_confidence_before:.2f}%"
)


print()
print("AFTER CALIBRATION")
print("-" * 70)

print(
    f"Temperature: "
    f"{temperature:.4f}"
)

print(
    f"Accuracy: "
    f"{calibrated_accuracy:.2f}%"
)

print(
    f"NLL: "
    f"{calibrated_loss:.4f}"
)

print(
    f"ECE: "
    f"{calibrated_ece:.4f}"
)

print(
    f"Average confidence on correct predictions: "
    f"{average_correct_confidence_after:.2f}%"
)

print(
    f"Average confidence on incorrect predictions: "
    f"{average_incorrect_confidence_after:.2f}%"
)

print(
    f"Maximum confidence on incorrect predictions: "
    f"{maximum_incorrect_confidence_after:.2f}%"
)


print()
print("CALIBRATION EFFECT")
print("-" * 70)

print(
    f"ECE change: "
    f"{uncalibrated_ece - calibrated_ece:.4f}"
)

print(
    f"NLL change: "
    f"{uncalibrated_loss - calibrated_loss:.4f}"
)

print(
    f"Accuracy change: "
    f"{calibrated_accuracy - uncalibrated_accuracy:.2f} percentage points"
)


print()
print("=" * 70)
print("FINAL CALIBRATED EVALUATION COMPLETE")
print("=" * 70)