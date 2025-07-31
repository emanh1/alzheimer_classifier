import os
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool, cpu_count
import torch

base_dir = os.path.join("..", "Data")
save_dir = os.path.join("..", "volumes") 
output_csv = os.path.join("..", "volume_metadata.csv")

def create_dataframe(base_dir=base_dir):
    pattern = re.compile(r"(OAS1_\d{4}_MR1_mpr-\d)")
    grouped = defaultdict(list)

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path): continue

        for fname in os.listdir(class_path):
            match = pattern.match(fname)
            if match:
                key = match.group(1)
                full_path = os.path.join(class_path, fname)
                grouped[key].append((full_path, class_name))

    records = []
    for key, items in grouped.items():
        for path, label in items:
            slice_num = int(re.search(r'_(\d+)\.', path).group(1))
            records.append({"group": key, "class": label, "path": path, "slice": slice_num})

    df = pd.DataFrame(records)
    df = df.sort_values(by=["group", "slice"])
    return df
    
def remap_label(label):
    return "Healthy" if label == "Non Demented" else "Alzheimer"

def process_group(args):
    group, group_df, save_dir = args
    label = remap_label(group_df["class"].iloc[0])
    paths = group_df.sort_values("slice")["path"].tolist()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    volume = []
    for path in paths:
        img = Image.open(path).convert("L")
        img = transform(img)
        volume.append(img)

    volume = torch.stack(volume, dim=1)  # (1, D, H, W)
    vol_path = os.path.join(save_dir, f"{group}.pt")
    torch.save(volume, vol_path)

    return {"group": group, "class": label, "path": vol_path}


def preprocess_and_save_volumes_parallel(df, save_dir=save_dir, num_workers=None):
    os.makedirs(save_dir, exist_ok=True)
    grouped = list(df.groupby("group"))

    job_args = [(group, group_df, save_dir) for group, group_df in grouped]

    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(f"Starting multiprocessing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_group, job_args), total=len(job_args), desc="Saving volumes"))

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = create_dataframe()
    volume_df = preprocess_and_save_volumes_parallel(df)
    volume_df.to_csv(output_csv, index=False)