from data.dataset import TomatoDataset


MANIFEST = "data/processed/tomato_manifest.csv"


dataset = TomatoDataset(
    MANIFEST,
    "train"
)

print("Number of training images:", len(dataset))

print("\nClass mapping:")
print(dataset.class_to_idx)

image, label = dataset[0]

print("\nFirst image:")
print("Tensor shape:", image.shape)
print("Label:", label)
print("Class:", dataset.idx_to_class[label])