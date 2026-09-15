from datasets import load_dataset

print("Loading PlantVillage split information...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)

for split_name in ["train", "test"]:

    paths = dataset[split_name]["text"]

    tomato_color_paths = [
        path for path in paths
        if path.startswith("raw/color/Tomato___")
    ]

    print(f"\n{split_name.upper()}")
    print(f"Total paths: {len(paths)}")
    print(f"Tomato color images: {len(tomato_color_paths)}")

    print("\nFirst 3 Tomato paths:")

    for path in tomato_color_paths[:3]:
        print(path)