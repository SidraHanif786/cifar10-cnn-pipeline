import os
import torch
import torchvision
import torchvision.transforms as transforms

RAW_DIR = "data/raw"

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root=RAW_DIR, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=RAW_DIR, train=False, download=True, transform=transform)
    torch.save(train_set, os.path.join(RAW_DIR, "train.pt"))
    torch.save(test_set, os.path.join(RAW_DIR, "test.pt"))
    print(f"Saved {len(train_set)} train and {len(test_set)} test samples to {RAW_DIR}")

if __name__ == "__main__":
    main()