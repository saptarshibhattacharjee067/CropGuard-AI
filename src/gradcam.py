from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_finetuned_tomato.pth"
)

IMAGE_SIZE = 224

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
    "healthy",
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    model = resnet18(weights=None)

    number_of_features = model.fc.in_features

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

    return model


transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_gradcam(image_path):
    model = load_model()

    activations = {}
    gradients = {}

    target_layer = model.layer4[-1].conv2

    def forward_hook(module, input_data, output):
        activations["value"] = output

        def save_gradient(gradient):
            gradients["value"] = gradient

        output.register_hook(save_gradient)

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        original_image
    )

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    model.zero_grad()

    output = model(image_tensor)

    predicted_class = output.argmax(
        dim=1
    ).item()

    target_score = output[
        0,
        predicted_class
    ]

    target_score.backward()

    hook.remove()

    activation = activations["value"]
    gradient = gradients["value"]

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activation
    ).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    if cam.max() != cam.min():
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )
    else:
        cam = np.zeros_like(cam)

    original_width, original_height = (
        original_image.size
    )

    cam_image = Image.fromarray(
        (cam * 255).astype(np.uint8)
    )

    cam_image = cam_image.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR
    )

    cam = np.array(cam_image) / 255.0

    heatmap = plt.get_cmap("jet")(
        cam
    )[:, :, :3]

    heatmap = (
        heatmap * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        heatmap
    )

    original_array = np.array(
        original_image
    ).astype(np.float32)

    heatmap_array = np.array(
        heatmap_image
    ).astype(np.float32)

    overlay = (
        0.5 * original_array
        + 0.5 * heatmap_array
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    overlay_image = Image.fromarray(
        overlay
    )

    predicted_label = CLASS_NAMES[
        predicted_class
    ]

    return (
        predicted_label,
        overlay_image
    )