from pathlib import Path
from collections import Counter

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

BATCH_SIZE = 32

CONFIDENCE_THRESHOLD = 60.0

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy"
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
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
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


total_images = 0

correct_predictions = 0

incorrect_predictions = 0


correct_confidences = []

incorrect_confidences = []

all_confidences = []

error_pairs = Counter()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = (
            torch.max(
                probabilities,
                dim=1
            )
        )


        for actual, predicted, confidence in zip(
            labels,
            predictions,
            confidences
        ):

            actual_class = actual.item()

            predicted_class = (
                predicted.item()
            )

            confidence_value = (
                confidence.item()
                * 100
            )


            total_images += 1

            all_confidences.append(
                confidence_value
            )


            if actual_class == predicted_class:

                correct_predictions += 1

                correct_confidences.append(
                    confidence_value
                )

            else:

                incorrect_predictions += 1

                incorrect_confidences.append(
                    confidence_value
                )


                pair = (
                    CLASS_NAMES[
                        actual_class
                    ],
                    CLASS_NAMES[
                        predicted_class
                    ]
                )

                error_pairs[pair] += 1


accuracy = (
    correct_predictions
    / total_images
    * 100
)


average_correct_confidence = (
    sum(correct_confidences)
    / len(correct_confidences)
)


average_incorrect_confidence = (
    sum(incorrect_confidences)
    / len(incorrect_confidences)
)


maximum_incorrect_confidence = max(
    incorrect_confidences
)


minimum_incorrect_confidence = min(
    incorrect_confidences
)


high_confidence_errors = [
    confidence
    for confidence in incorrect_confidences
    if confidence >= CONFIDENCE_THRESHOLD
]


high_confidence_error_count = len(
    high_confidence_errors
)


high_confidence_error_percentage = (
    high_confidence_error_count
    / incorrect_predictions
    * 100
)


correct_above_threshold = sum(
    confidence >= CONFIDENCE_THRESHOLD
    for confidence in correct_confidences
)


correct_above_threshold_percentage = (
    correct_above_threshold
    / correct_predictions
    * 100
)


print()

print("=" * 65)

print(
    "FINE-TUNED RESNET18 ERROR & CONFIDENCE ANALYSIS"
)

print("=" * 65)


print()

print(
    f"Total test images: "
    f"{total_images}"
)

print(
    f"Correct predictions: "
    f"{correct_predictions}"
)

print(
    f"Incorrect predictions: "
    f"{incorrect_predictions}"
)

print(
    f"Accuracy: "
    f"{accuracy:.2f}%"
)


print()

print(
    "CONFIDENCE ANALYSIS"
)

print(
    "-" * 65
)


print(
    f"Average confidence on correct predictions: "
    f"{average_correct_confidence:.2f}%"
)

print(
    f"Average confidence on incorrect predictions: "
    f"{average_incorrect_confidence:.2f}%"
)

print(
    f"Maximum confidence on an incorrect prediction: "
    f"{maximum_incorrect_confidence:.2f}%"
)

print(
    f"Minimum confidence on an incorrect prediction: "
    f"{minimum_incorrect_confidence:.2f}%"
)


print()

print(
    f"Current screening threshold: "
    f"{CONFIDENCE_THRESHOLD:.0f}%"
)

print(
    f"Incorrect predictions above threshold: "
    f"{high_confidence_error_count} "
    f"out of {incorrect_predictions}"
)

print(
    f"Percentage of errors above threshold: "
    f"{high_confidence_error_percentage:.2f}%"
)


print()

print(
    f"Correct predictions above threshold: "
    f"{correct_above_threshold} "
    f"out of {correct_predictions}"
)

print(
    f"Percentage of correct predictions above threshold: "
    f"{correct_above_threshold_percentage:.2f}%"
)


print()

print(
    "MOST COMMON PREDICTION ERRORS"
)

print(
    "-" * 65
)


if len(error_pairs) == 0:

    print(
        "No incorrect predictions found."
    )

else:

    for (
        actual_class,
        predicted_class
    ), count in error_pairs.most_common():

        percentage = (
            count
            / incorrect_predictions
            * 100
        )

        print(
            f"{actual_class:<42}"
            f" -> "
            f"{predicted_class:<42}"
            f": {count:>3}"
            f" ({percentage:.2f}% of errors)"
        )


print()

print(
    "ERROR SUMMARY BY ACTUAL CLASS"
)

print(
    "-" * 65
)


actual_error_counts = Counter()


for (
    actual_class,
    predicted_class
), count in error_pairs.items():

    actual_error_counts[
        actual_class
    ] += count


for class_name in CLASS_NAMES:

    count = actual_error_counts[
        class_name
    ]

    print(
        f"{class_name:<45}: {count}"
    )


print()

print(
    "ERROR & CONFIDENCE ANALYSIS COMPLETE."
)