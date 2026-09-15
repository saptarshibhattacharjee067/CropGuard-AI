from datasets import load_dataset
from pathlib import Path
import csv
import random

print("Loading PlantVillage split information...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)

DATA_DIR = Path("data/raw/color")

records = []

# 1. Collect official train/test images

for split_name in ["train", "test"]:

    paths = dataset[split_name]["text"]

    for path in paths:

        if not path.startswith("raw/color/Tomato___"):
            continue

        relative_path = Path(
            path.replace("raw/color/", "")
        )

        local_path = DATA_DIR / relative_path

        if not local_path.exists():
            print(f"WARNING: File not found: {local_path}")
            continue

        class_name = relative_path.parts[0].replace(
            "Tomato___", ""
        )

        # Filename contains the leaf identifier
        filename = relative_path.name
        leaf_id = filename.split("__")[0]

        records.append({
            "path": str(local_path).replace("\\", "/"),
            "class": class_name,
            "split": split_name,
            "leaf_id": leaf_id
        })

# 2. Create validation split WITHOUT leaf leakage

train_records = [
    record for record in records
    if record["split"] == "train"
]

random.seed(42)

classes = sorted(
    set(record["class"] for record in train_records)
)

for class_name in classes:

    class_records = [
        record
        for record in train_records
        if record["class"] == class_name
    ]

    # Find unique leaves
    leaf_groups = {}

    for record in class_records:
        leaf_groups.setdefault(
            record["leaf_id"], []
        ).append(record)

    leaf_ids = list(leaf_groups.keys())

    random.shuffle(leaf_ids)

    validation_leaf_count = int(
        len(leaf_ids) * 0.20
    )

    validation_leaf_ids = set(
        leaf_ids[:validation_leaf_count]
    )

    for leaf_id in validation_leaf_ids:

        for record in leaf_groups[leaf_id]:
            record["split"] = "validation"

# 3. Create manifest

output_path = Path(
    "data/processed/tomato_manifest.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "path",
            "class",
            "split",
            "leaf_id"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

# 4. Print final results

print("\nFinal dataset split")
print("-------------------")

for split_name in [
    "train",
    "validation",
    "test"
]:

    count = sum(
        1
        for record in records
        if record["split"] == split_name
    )

    print(
        f"{split_name.capitalize():12}: {count}"
    )

print(f"\nTotal images: {len(records)}")

print("\nManifest created successfully!")
print(f"Saved to: {output_path}")