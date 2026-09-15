import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.transforms import functional as TF

from PIL import ImageEnhance

from data.dataset import TomatoDataset


# CONFIGURATION

MANIFEST = "data/processed/tomato_manifest.csv"
MODEL_PATH = "models/resnet18_finetuned_tomato.pth"

BATCH_SIZE = 32
NUM_CLASSES = 10


# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# LOAD TEST DATASET

test_dataset = TomatoDataset(
    MANIFEST,
    "test",
    use_augmentation=False
)

print("Test images:", len(test_dataset))


# LOAD FINE-TUNED RESNET18

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

# NORMALIZATION

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# TRANSFORMATIONS

# Normal / clean images
clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


# Darker images
dark_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(
        lambda image:
        ImageEnhance.Brightness(image).enhance(0.5)
    ),
    transforms.ToTensor(),
    normalize
])


# Higher contrast
contrast_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(
        lambda image:
        ImageEnhance.Contrast(image).enhance(1.5)
    ),
    transforms.ToTensor(),
    normalize
])


# Rotated images
rotation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(
        lambda image:
        TF.rotate(image, 20)
    ),
    transforms.ToTensor(),
    normalize
])


# Blurred images
blur_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(
        kernel_size=5,
        sigma=2.0
    ),
    transforms.ToTensor(),
    normalize
])


# EVALUATION FUNCTION

def evaluate_condition(
    name,
    transform
):

    # Replace dataset transformation
    test_dataset.transform = transform

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        correct / total
    ) * 100

    print(
        f"{name:<20} Accuracy: {accuracy:.2f}%"
    )

    return accuracy


# RUN ROBUSTNESS TESTS

print()
print("=" * 55)
print("CROPGUARD AI - ROBUSTNESS TEST")
print("=" * 55)

results = {}

results["Clean"] = evaluate_condition(
    "Clean images",
    clean_transform
)

results["Dark"] = evaluate_condition(
    "Dark images",
    dark_transform
)

results["High Contrast"] = evaluate_condition(
    "High contrast",
    contrast_transform
)

results["Rotated"] = evaluate_condition(
    "Rotated images",
    rotation_transform
)

results["Blurred"] = evaluate_condition(
    "Blurred images",
    blur_transform
)


# SUMMARY

print()
print("=" * 55)
print("ROBUSTNESS SUMMARY")
print("=" * 55)

for condition, accuracy in results.items():

    print(
        f"{condition:<20} {accuracy:.2f}%"
    )


# FIND WORST CONDITION

worst_condition = min(
    results,
    key=results.get
)

worst_accuracy = results[
    worst_condition
]

print()
print(
    f"Worst condition: {worst_condition}"
)

print(
    f"Worst accuracy: {worst_accuracy:.2f}%"
)

print()
print("Robustness testing complete!")