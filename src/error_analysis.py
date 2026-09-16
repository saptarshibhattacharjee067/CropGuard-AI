from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib.pyplot as plt

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

CSV_PATH = (
    RESULTS_DIR
    / "resnet18_error_analysis.csv"
)

FIGURE_PATH = (
    RESULTS_DIR
    / "resnet18_misclassified_images.png"
)

BATCH_SIZE = 32

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


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
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


misclassified_images = []

error_records = []

test_index = 0


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

        for i in range(
            len(labels)
        ):

            actual_label = (
                labels[i].item()
            )

            predicted_label = (
                predictions[i].item()
            )

            confidence = (
                confidences[i].item()
                * 100
            )


            if actual_label != predicted_label:

                actual_name = (
                    CLASS_NAMES[
                        actual_label
                    ]
                )

                predicted_name = (
                    CLASS_NAMES[
                        predicted_label
                    ]
                )


                error_records.append({
                    "test_index": test_index,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "confidence": confidence
                })


                misclassified_images.append(
                    (
                        images[i].cpu(),
                        actual_label,
                        predicted_label,
                        confidence
                    )
                )


            test_index += 1


print()
print(
    "Total test images:",
    test_index
)

print(
    "Total misclassified images:",
    len(misclassified_images)
)


error_dataframe = pd.DataFrame(
    error_records
)


error_dataframe = (
    error_dataframe
    .sort_values(
        "confidence",
        ascending=False
    )
)


error_dataframe.to_csv(
    CSV_PATH,
    index=False
)


print()
print(
    "Error analysis saved to:"
)

print(
    CSV_PATH
)


if len(misclassified_images) > 0:

    number_to_display = min(
        len(misclassified_images),
        54
    )

    columns = 6

    rows = (
        number_to_display
        + columns
        - 1
    ) // columns


    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 3 * rows)
    )


    axes = axes.flatten()


    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406
        ]
    ).view(
        3,
        1,
        1
    )


    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225
        ]
    ).view(
        3,
        1,
        1
    )


    for i in range(
        number_to_display
    ):

        image = (
            misclassified_images[i][0]
        )

        actual_label = (
            misclassified_images[i][1]
        )

        predicted_label = (
            misclassified_images[i][2]
        )

        confidence = (
            misclassified_images[i][3]
        )


        image = (
            image * std
            + mean
        )


        image = image.clamp(
            0,
            1
        )


        image = image.permute(
            1,
            2,
            0
        )


        axes[i].imshow(
            image
        )


        actual_name = (
            CLASS_NAMES[
                actual_label
            ]
        )

        predicted_name = (
            CLASS_NAMES[
                predicted_label
            ]
        )


        axes[i].set_title(
            "Actual: "
            + actual_name
            + "\nPredicted: "
            + predicted_name
            + f"\nConfidence: {confidence:.2f}%",
            fontsize=8
        )


        axes[i].axis(
            "off"
        )


    for i in range(
        number_to_display,
        len(axes)
    ):

        axes[i].axis(
            "off"
        )


    plt.suptitle(
        "Fine-Tuned ResNet18 Misclassified Images",
        fontsize=16
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )


    plt.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )


    print()
    print(
        "Misclassified image figure saved to:"
    )

    print(
        FIGURE_PATH
    )


    plt.show()


print()
print(
    "Top Error Cases"
)

print(
    "---------------"
)


if len(error_dataframe) > 0:

    print(
        error_dataframe.head(
            20
        ).to_string(
            index=False
        )
    )

else:

    print(
        "No misclassified images found."
    )