import os
import yaml
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

class CNN(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters, num_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]

def main():
    params = load_params()
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_set = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=False)
    val_set = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=False)
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(params["num_filters"], params["dropout_rate"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(params["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        history.append({"epoch": epoch + 1, "val_loss": val_loss / len(val_loader), "val_acc": val_acc})
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "model.pth"))
    with open(os.path.join(MODELS_DIR, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

if __name__ == "__main__":
    main()