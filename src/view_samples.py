from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path("data/raw/color")

# Get all tomato class folders
class_folders = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir() and folder.name.startswith("Tomato___")
)

# Take one image from each class
samples = []

for folder in class_folders:
    image_files = list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))

    if image_files:
        samples.append((folder.name, image_files[0]))


# Create the figure
plt.figure(figsize=(15, 12))

for index, (class_name, image_path) in enumerate(samples):

    image = Image.open(image_path)

    plt.subplot(4, 3, index + 1)
    plt.imshow(image)
    plt.title(class_name.replace("Tomato___", "").replace("_", " "))
    plt.axis("off")

plt.tight_layout()
plt.show()