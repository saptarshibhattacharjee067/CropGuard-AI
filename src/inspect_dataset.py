from datasets import load_dataset

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)

print("\nDataset:")
print(dataset)

print("\nTrain features:")
print(dataset["train"].features)

print("\nFirst training example:")
print(dataset["train"][0])