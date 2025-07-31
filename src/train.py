import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from pathlib import Path

from model import Tiny3DCNN
from dataset import CachedVolumeDataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "volumes"
CSV_PATH = BASE_DIR / "volume_metadata.csv"
TRAIN_CSV = BASE_DIR / "train_volume.csv"
VAL_CSV = BASE_DIR / "val_volume.csv"
MODEL_PATH = BASE_DIR / "best_model.pt"
CM_PATH = BASE_DIR / "confusion_matrix.png"


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    class_names = sorted(df["class"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)

    train_dataset = CachedVolumeDataset(TRAIN_CSV, class_to_idx)
    val_dataset = CachedVolumeDataset(VAL_CSV, class_to_idx)

    return train_dataset, val_dataset, class_names, class_to_idx


def compute_weights(train_loader, device):
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())

    classes = np.unique(all_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_model():
    mlflow.set_experiment("Alzheimer_MRI_Classifier")
    with mlflow.start_run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset, val_dataset, class_names, _ = prepare_data(CSV_PATH)
        train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=6, num_workers=4, pin_memory=True)

        model = Tiny3DCNN(in_channels=1, n_classes=len(class_names)).to(device)
        weights = compute_weights(train_loader, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        mlflow.log_params({"batch_size": 6, "lr": 1e-3, "patience": 5})

        best_val_loss = float('inf')
        patience, counter = 5, 0
        train_losses, val_losses = [], []
        scaler = torch.amp.GradScaler()

        for epoch in range(100):
            model.train()
            train_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device_type=device.type):
                        out = model(x)
                        loss = criterion(out, y)
                    val_loss += loss.item()
                    preds = torch.argmax(out, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                mlflow.pytorch.log_model(model, "model")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

        # Final evaluation
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        mlflow.log_metrics({f"f1_{cls}": report[cls]['f1-score'] for cls in class_names})

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues")
        fig.savefig(CM_PATH)
        mlflow.log_artifact(str(CM_PATH))


if __name__ == "__main__":
    train_model()
