from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


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
    "healthy"
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_model():

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

    return model


transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


def generate_gradcam(
    image_path,
    output_path
):

    model = load_model()

    activations = []
    gradients = []

    target_layer = (
        model.layer4[-1]
    )

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output
        )

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        original_image = (
            Image.open(
                image_path
            ).convert("RGB")
        )

        image_tensor = transform(
            original_image
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(device)
        )

        model.zero_grad()

        output = model(
            image_tensor
        )

        predicted_class = (
            torch.argmax(
                output,
                dim=1
            ).item()
        )

        target_score = (
            output[0, predicted_class]
        )

        target_score.backward()

        feature_maps = (
            activations[0]
            .detach()
            .cpu()
        )

        gradient_maps = (
            gradients[0]
            .detach()
            .cpu()
        )

        weights = (
            gradient_maps.mean(
                dim=(2, 3),
                keepdim=True
            )
        )

        cam = (
            weights * feature_maps
        ).sum(
            dim=1
        ).squeeze(0)

        cam = torch.relu(
            cam
        )

        cam = cam.numpy()

        if cam.max() > 0:

            cam = (
                cam / cam.max()
            )

        original_width, original_height = (
            original_image.size
        )

        cam_image = Image.fromarray(
            np.uint8(
                cam * 255
            )
        )

        cam_image = (
            cam_image.resize(
                (
                    original_width,
                    original_height
                ),
                Image.Resampling.BILINEAR
            )
        )

        cam = np.asarray(
            cam_image
        ) / 255.0

        original_array = (
            np.asarray(
                original_image
            ) / 255.0
        )

        figure = plt.figure(
            figsize=(10, 5)
        )

        ax1 = figure.add_subplot(
            1,
            2,
            1
        )

        ax1.imshow(
            original_array
        )

        ax1.set_title(
            "Uploaded Image"
        )

        ax1.axis(
            "off"
        )

        ax2 = figure.add_subplot(
            1,
            2,
            2
        )

        ax2.imshow(
            original_array
        )

        ax2.imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        ax2.set_title(
            "Model Attention — Grad-CAM"
        )

        ax2.axis(
            "off"
        )

        figure.suptitle(
            "CropGuard AI Grad-CAM Explanation",
            fontsize=14
        )

        plt.tight_layout()

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(
            figure
        )

        return {
            "class": CLASS_NAMES[
                predicted_class
            ],
            "output_path": str(
                output_path
            )
        }

    finally:

        forward_handle.remove()

        backward_handle.remove()


if __name__ == "__main__":

    print(
        "CropGuard AI Grad-CAM"
    )

    print(
        "Using device:",
        device
    )

    image_path = input(
        "Enter image path: "
    )

    if not Path(
        image_path
    ).exists():

        print(
            "Error: Image file not found."
        )

    else:

        output_path = (
            PROJECT_ROOT
            / "results"
            / "gradcam_test.png"
        )

        result = generate_gradcam(
            image_path,
            output_path
        )

        print()

        print(
            "Predicted class:",
            result["class"]
        )

        print(
            "Grad-CAM saved to:",
            result["output_path"]
        )