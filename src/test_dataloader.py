from data.dataset import TomatoDataset
from torch.utils.data import DataLoader


MANIFEST = "data/processed/tomato_manifest.csv"


# Create datasets
train_dataset = TomatoDataset(
    MANIFEST,
    "train"
)

validation_dataset = TomatoDataset(
    MANIFEST,
    "validation"
)

test_dataset = TomatoDataset(
    MANIFEST,
    "test"
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# Get one batch
images, labels = next(iter(train_loader))


print("Dataset sizes")
print("--------------------")
print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))

print("\nFirst training batch")
print("--------------------")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

print("\nLabels in first batch:")
print(labels)