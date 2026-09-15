import torch
from torch.utils.data import DataLoader
from collections import Counter

from data.dataset import TomatoDataset
from models.cnn import TomatoCNN


# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"
MODEL_PATH = "models/tomato_cnn.pth"

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


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load test dataset

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


# Load trained model

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


# Collect prediction errors

errors = Counter()

total_images = 0
correct = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for actual, predicted in zip(
            labels,
            predictions.cpu()
        ):

            total_images += 1

            if actual == predicted:

                correct += 1

            else:

                actual_name = CLASS_NAMES[
                    actual.item()
                ]

                predicted_name = CLASS_NAMES[
                    predicted.item()
                ]

                errors[
                    (actual_name, predicted_name)
                ] += 1


# Results

print("\n" + "=" * 50)

print("ERROR ANALYSIS")

print("=" * 50)

print(
    f"Total test images: {total_images}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Incorrect predictions: "
    f"{total_images - correct}"
)

print(
    f"Accuracy: "
    f"{correct / total_images * 100:.2f}%"
)


# Most common mistakes

print("\nMost common prediction errors:")
print("-" * 50)

for (actual, predicted), count in errors.most_common():

    print(
        f"{actual:35} -> "
        f"{predicted:35} : {count}"
    )