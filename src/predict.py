import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

DISEASE_INFO_PATH = (
    PROJECT_ROOT
    / "data"
    / "disease_info.json"
)


IMAGE_SIZE = 224

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
    "healthy",
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


if TEMPERATURE_PATH.exists():

    with open(
        TEMPERATURE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        calibration_data = json.load(
            file
        )

    TEMPERATURE = float(
        calibration_data[
            "temperature"
        ]
    )

else:

    TEMPERATURE = 1.0


transform = transforms.Compose([
    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
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


def predict_image(
    image_path,
    top_k=3
):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    )


    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        calibrated_outputs = (
            outputs
            / TEMPERATURE
        )


        probabilities = torch.softmax(
            calibrated_outputs,
            dim=1
        )


    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=top_k,
            dim=1
        )
    )


    predictions = []


    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        class_name = (
            CLASS_NAMES[
                index.item()
            ]
        )


        confidence = (
            probability.item()
            * 100
        )


        predictions.append({
            "class": class_name,
            "confidence": confidence
        })


    return predictions


def load_disease_info():

    if not DISEASE_INFO_PATH.exists():

        return {}


    with open(
        DISEASE_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def get_prediction_status(
    confidence
):

    if confidence >= CONFIDENCE_THRESHOLD:

        return (
            "sufficiently_confident",
            "The model produced a prediction above the current screening threshold."
        )

    return (
        "uncertain",
        "The model confidence is below the current screening threshold. Consider uploading a clearer image or seeking agricultural guidance."
    )


if __name__ == "__main__":

    print()

    print(
        "CropGuard AI Predictor"
    )

    print(
        "======================"
    )

    print(
        f"Calibration temperature: "
        f"{TEMPERATURE:.4f}"
    )

    print(
        f"Screening threshold: "
        f"{CONFIDENCE_THRESHOLD:.0f}%"
    )

    print()

    image_path = input(
        "Enter image path: "
    )


    image_path = Path(
        image_path
    )


    if not image_path.exists():

        print()

        print(
            "Error: Image file not found."
        )

    else:

        predictions = predict_image(
            image_path,
            top_k=3
        )


        print()

        print(
            "Top Predictions"
        )

        print(
            "----------------"
        )


        for position, prediction in enumerate(
            predictions,
            start=1
        ):

            disease = (
                prediction["class"]
            )

            confidence = (
                prediction["confidence"]
            )


            print(
                f"{position}. "
                f"{disease} → "
                f"{confidence:.2f}%"
            )


        predicted_disease = (
            predictions[0]["class"]
        )

        predicted_confidence = (
            predictions[0]["confidence"]
        )


        status, message = (
            get_prediction_status(
                predicted_confidence
            )
        )


        print()

        print(
            "Prediction Status"
        )

        print(
            "-----------------"
        )

        print(
            message
        )


        disease_info = (
            load_disease_info()
        )


        info = disease_info.get(
            predicted_disease
        )


        if info:

            print()

            print(
                "=" * 50
            )

            print(
                "DISEASE INFORMATION"
            )

            print(
                "=" * 50
            )


            print(
                f"Name: "
                f"{info['name']}"
            )

            print(
                f"Type: "
                f"{info['type']}"
            )


            print()

            print(
                "Description:"
            )

            print(
                info["description"]
            )


            print()

            print(
                "Symptoms:"
            )


            for symptom in info[
                "symptoms"
            ]:

                print(
                    f"- {symptom}"
                )


            print()

            print(
                "Management:"
            )


            for item in info[
                "management"
            ]:

                print(
                    f"- {item}"
                )


            print()

            print(
                "Prevention:"
            )


            for item in info[
                "prevention"
            ]:

                print(
                    f"- {item}"
                )