import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.models import resnet18

from data.dataset import TomatoDataset

# Configuration

MANIFEST = "data/processed/tomato_manifest.csv"

MODEL_PATH = "models/resnet18_finetuned_tomato.pth"

OUTPUT_PATH = "results/gradcam_resnet18.png"

NUM_CLASSES = 10


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load dataset

test_dataset = TomatoDataset(
    MANIFEST,
    "test",
    use_augmentation=False
)


# Load model

model = resnet18(
    weights=None
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


# Grad-CAM setup

activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


# Use the final convolutional layer
target_layer = model.layer4[-1]

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# Select an image

image, true_label = test_dataset[0]

image_batch = image.unsqueeze(0).to(device)


# Forward pass

model.zero_grad()

output = model(image_batch)

predicted_label = torch.argmax(
    output,
    dim=1
).item()


# Backward pass

score = output[0, predicted_label]

score.backward()


# Generate Grad-CAM

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (
    weights * activations
).sum(
    dim=1
).squeeze()


cam = torch.relu(cam)

cam = cam.detach().cpu().numpy()


# Normalize heatmap
cam = (
    cam - cam.min()
) / (
    cam.max() - cam.min() + 1e-8
)


# Prepare original image

original_image = image.permute(
    1, 2, 0
).cpu().numpy()


# Undo ImageNet normalization
mean = np.array(
    [0.485, 0.456, 0.406]
)

std = np.array(
    [0.229, 0.224, 0.225]
)

original_image = (
    original_image * std
) + mean

original_image = np.clip(
    original_image,
    0,
    1
)


# Resize heatmap

heatmap = Image.fromarray(
    np.uint8(cam * 255)
)

heatmap = heatmap.resize(
    (224, 224)
)

heatmap = np.array(
    heatmap
) / 255.0


# Create visualization

plt.figure(
    figsize=(15, 5)
)


# Original image
plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    original_image
)

plt.title(
    f"Original\nActual: "
    f"{test_dataset.idx_to_class[true_label]}"
)

plt.axis("off")


# Heatmap
plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    f"Grad-CAM\nPredicted: "
    f"{test_dataset.idx_to_class[predicted_label]}"
)

plt.axis("off")


# Overlay
plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    original_image
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)

plt.title(
    "Model Attention"
)

plt.axis("off")


# Save result

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print()
print("=" * 50)
print("Grad-CAM COMPLETE")
print("=" * 50)

print(
    "Actual class:",
    test_dataset.idx_to_class[true_label]
)

print(
    "Predicted class:",
    test_dataset.idx_to_class[predicted_label]
)

print(
    "Saved to:",
    OUTPUT_PATH
)