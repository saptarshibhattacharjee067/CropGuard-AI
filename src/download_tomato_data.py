from huggingface_hub import hf_hub_download
from zipfile import ZipFile
from pathlib import Path

print("Downloading PlantVillage image archive...")
print("This may take some time because the archive is about 2 GB.")

zip_path = hf_hub_download(
    repo_id="mohanty/PlantVillage",
    filename="data.zip",
    repo_type="dataset"
)

print("\nDownload complete!")
print("Extracting Tomato images...")

output_dir = Path("data/raw/color")
output_dir.mkdir(parents=True, exist_ok=True)

with ZipFile(zip_path, "r") as zip_file:

    tomato_files = [
        name
        for name in zip_file.namelist()
        if name.startswith("raw/color/Tomato___")
        and name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(tomato_files)} Tomato images.")

    for file_name in tomato_files:
        zip_file.extract(file_name, "data/raw")

print("\nTomato images extracted successfully!")
# Extract PlantVillage's official train/test split files
split_files = [
    "color_train.txt",
    "color_test.txt"
]

with ZipFile(zip_path, "r") as zip_file:

    for split_file in split_files:
        matching_files = [
            name for name in zip_file.namelist()
            if name.endswith(split_file)
        ]

        if matching_files:
            zip_file.extract(
                matching_files[0],
                "data/raw"
            )
            print(f"Extracted {split_file}")
print("\nLooking for split files...")

with ZipFile(zip_path, "r") as zip_file:
    for name in zip_file.namelist():
        if name.lower().endswith(".txt"):
            print(name)