from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TomatoDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        split,
        use_augmentation=False
    ):

        # Load our CSV manifest
        self.data = pd.read_csv(manifest_path)

        # Keep only the requested split
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        # Create a fixed mapping from class name -> number
        classes = sorted(
            self.data["class"].unique()
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(classes)
        }

        self.idx_to_class = {
            index: class_name
            for class_name, index
            in self.class_to_idx.items()
        }

        # Image preprocessing
        if use_augmentation and split == "train":

            self.transform = transforms.Compose([

                transforms.Resize((224, 224)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomRotation(15),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transforms.Compose([

                transforms.Resize((224, 224)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # Get image path and class
        row = self.data.iloc[index]

        image_path = Path(row["path"])

        class_name = row["class"]

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply preprocessing
        image = self.transform(image)

        # Convert class name to numerical label
        label = self.class_to_idx[class_name]

        return image, label