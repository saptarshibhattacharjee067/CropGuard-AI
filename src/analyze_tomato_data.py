from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/color")

print("Analyzing Tomato dataset...\n")

# Find all class folders
class_folders = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir() and folder.name.startswith("Tomato___")
)

print(f"Number of classes: {len(class_folders)}\n")

total_images = 0

for folder in class_folders:

    image_files = list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))

    print(f"{folder.name}: {len(image_files)} images")

    total_images += len(image_files)

print(f"\nTotal images: {total_images}")