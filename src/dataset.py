import torch
import pandas as pd
from torch.utils.data import Dataset

class CachedVolumeDataset(Dataset):
    def __init__(self, csv_path, class_to_idx):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume = torch.load(row["path"])  # (1, D, H, W)
        label = self.class_to_idx[row["class"]]
        return volume, label
