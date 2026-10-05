import os
import yaml
import torch
from torch.utils.data import random_split
from torchvision import transforms

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["preprocess"]

def main():
    params = load_params()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_set = torch.load(os.path.join(RAW_DIR, "train.pt"), weights_only=False)
    test_set = torch.load(os.path.join(RAW_DIR, "test.pt"), weights_only=False)

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set.transform = transform
    test_set.transform = transform

    val_size = int(len(train_set) * params["val_size"])
    train_size = len(train_set) - val_size
    generator = torch.Generator().manual_seed(params["seed"])
    train_subset, val_subset = random_split(train_set, [train_size, val_size], generator=generator)

    torch.save(train_subset, os.path.join(PROCESSED_DIR, "train.pt"))
    torch.save(val_subset, os.path.join(PROCESSED_DIR, "val.pt"))
    torch.save(test_set, os.path.join(PROCESSED_DIR, "test.pt"))
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_set)}")

if __name__ == "__main__":
    main()