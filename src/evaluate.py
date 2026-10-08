import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from train import CNN, load_params as load_train_params

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

def main():
    params = load_train_params()
    test_set = torch.load(f"{PROCESSED_DIR}/test.pt", weights_only=False)
    test_loader = DataLoader(test_set, batch_size=params["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(params["num_filters"], params["dropout_rate"]).to(device)
    model.load_state_dict(torch.load(f"{MODELS_DIR}/model.pth", map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    metrics = {"test_loss": total_loss / len(test_loader), "test_accuracy": test_acc}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("models/confusion_matrix.png")
    print(metrics)

if __name__ == "__main__":
    main()