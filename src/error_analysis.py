import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

# Dataset

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


# Find misclassified images

misclassified_images = []
misclassified_actual = []
misclassified_predicted = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):

            if predictions[i].cpu() != labels[i]:

                misclassified_images.append(
                    images[i].cpu()
                )

                misclassified_actual.append(
                    labels[i].item()
                )

                misclassified_predicted.append(
                    predictions[i].item()
                )

            # Stop after collecting 20 errors
            if len(misclassified_images) >= 20:
                break

        if len(misclassified_images) >= 20:
            break


# Display errors

print(
    f"Showing {len(misclassified_images)} misclassified images."
)


fig, axes = plt.subplots(
    4,
    5,
    figsize=(15, 12)
)


for i, ax in enumerate(axes.flat):

    image = misclassified_images[i]

    image = image.permute(
        1,
        2,
        0
    )

    image = image.clamp(
        0,
        1
    )

    ax.imshow(image)

    actual = CLASS_NAMES[
        misclassified_actual[i]
    ]

    predicted = CLASS_NAMES[
        misclassified_predicted[i]
    ]

    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}",
        fontsize=8
    )

    ax.axis("off")


plt.tight_layout()

plt.show()