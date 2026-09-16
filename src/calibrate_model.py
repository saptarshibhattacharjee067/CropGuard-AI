from pathlib import Path
import json

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TEMPERATURE_PATH = (
    RESULTS_DIR
    / "calibration_temperature.json"
)

CALIBRATION_PLOT_PATH = (
    RESULTS_DIR
    / "reliability_diagram.png"
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


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


validation_dataset = TomatoDataset(
    str(MANIFEST),
    "validation"
)

validation_loader = DataLoader(
    validation_dataset,
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
    "Collecting validation predictions..."
)


with torch.no_grad():

    for images, labels in validation_loader:

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


print(
    "Validation images:",
    len(labels)
)


criterion = nn.CrossEntropyLoss()


nll_before = criterion(
    logits,
    labels
).item()


predictions_before = (
    logits.argmax(
        dim=1
    )
)


accuracy_before = (
    (
        predictions_before == labels
    )
    .float()
    .mean()
    .item()
    * 100
)


temperature = torch.nn.Parameter(
    torch.ones(
        1
    )
)


optimizer = torch.optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=100,
    line_search_fn="strong_wolfe"
)


def closure():

    optimizer.zero_grad()

    scaled_logits = (
        logits
        / temperature
    )

    loss = criterion(
        scaled_logits,
        labels
    )

    loss.backward()

    return loss


print()
print(
    "Learning temperature..."
)


optimizer.step(
    closure
)


learned_temperature = (
    temperature.detach()
    .item()
)


if learned_temperature <= 0:

    learned_temperature = 1.0


scaled_logits = (
    logits
    / learned_temperature
)


nll_after = criterion(
    scaled_logits,
    labels
).item()


predictions_after = (
    scaled_logits.argmax(
        dim=1
    )
)


accuracy_after = (
    (
        predictions_after == labels
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


    ece = torch.tensor(
        0.0
    )


    bin_data = []


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

            bin_data.append({
                "confidence": 0.0,
                "accuracy": 0.0,
                "count": 0
            })

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


        bin_data.append({
            "confidence": bin_confidence,
            "accuracy": bin_accuracy,
            "count": count
        })


    return (
        ece.item(),
        bin_data
    )


ece_before, bins_before = (
    calculate_ece(
        logits,
        labels,
        NUMBER_OF_BINS
    )
)


ece_after, bins_after = (
    calculate_ece(
        scaled_logits,
        labels,
        NUMBER_OF_BINS
    )
)


print()
print("=" * 65)

print(
    "CROPGUARD AI CONFIDENCE CALIBRATION"
)

print("=" * 65)


print()

print(
    f"Validation images: "
    f"{len(labels)}"
)


print()

print(
    "BEFORE TEMPERATURE SCALING"
)

print(
    "-" * 65
)

print(
    f"Accuracy: "
    f"{accuracy_before:.2f}%"
)

print(
    f"Cross-Entropy / NLL: "
    f"{nll_before:.4f}"
)

print(
    f"Expected Calibration Error (ECE): "
    f"{ece_before:.4f}"
)


print()

print(
    "LEARNED TEMPERATURE"
)

print(
    "-" * 65
)

print(
    f"Temperature: "
    f"{learned_temperature:.4f}"
)


print()

print(
    "AFTER TEMPERATURE SCALING"
)

print(
    "-" * 65
)

print(
    f"Accuracy: "
    f"{accuracy_after:.2f}%"
)

print(
    f"Cross-Entropy / NLL: "
    f"{nll_after:.4f}"
)

print(
    f"Expected Calibration Error (ECE): "
    f"{ece_after:.4f}"
)


print()

print(
    "CALIBRATION CHANGE"
)

print(
    "-" * 65
)

print(
    f"ECE improvement: "
    f"{ece_before - ece_after:.4f}"
)

print(
    f"NLL improvement: "
    f"{nll_before - nll_after:.4f}"
)


calibration_data = {
    "temperature": learned_temperature,
    "validation_images": len(labels),
    "accuracy_before": accuracy_before,
    "accuracy_after": accuracy_after,
    "nll_before": nll_before,
    "nll_after": nll_after,
    "ece_before": ece_before,
    "ece_after": ece_after
}


with open(
    TEMPERATURE_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        calibration_data,
        file,
        indent=4
    )


print()

print(
    "Calibration parameters saved to:"
)

print(
    TEMPERATURE_PATH
)


bin_centers = [
    (i + 0.5) / NUMBER_OF_BINS
    for i in range(
        NUMBER_OF_BINS
    )
]


confidence_before = []

accuracy_before_bins = []

confidence_after = []

accuracy_after_bins = []


for i in range(
    NUMBER_OF_BINS
):

    confidence_before.append(
        bins_before[i]["confidence"]
    )

    accuracy_before_bins.append(
        bins_before[i]["accuracy"]
    )

    confidence_after.append(
        bins_after[i]["confidence"]
    )

    accuracy_after_bins.append(
        bins_after[i]["accuracy"]
    )


plt.figure(
    figsize=(8, 7)
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)


plt.plot(
    confidence_before,
    accuracy_before_bins,
    marker="o",
    label="Before calibration"
)


plt.plot(
    confidence_after,
    accuracy_after_bins,
    marker="o",
    label="After calibration"
)


plt.xlabel(
    "Confidence"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "CropGuard AI Reliability Diagram"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    CALIBRATION_PLOT_PATH,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


print()

print(
    "Reliability diagram saved to:"
)

print(
    CALIBRATION_PLOT_PATH
)


print()

print(
    "Confidence calibration complete."
)