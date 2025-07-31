# Alzheimer MRI Volume Classifier
![Brain](.github/brain.gif)

This project implements a deep learning pipeline for classifying Alzheimer's disease from 3D MRI brain scan volumes, with experiment tracking using MLflow. The workflow covers data preprocessing, volume construction, model training, evaluation, visualization, and experiment management.

## Dataset
- [OASIS Alzheimer's Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)

## Project Structure

- `src/` — Source code (data, model, training, visualization)
- `notebooks/volume_classifier.ipynb` — Main notebook for workflow
- `volume_metadata.csv`, `train_volume.csv`, `val_volume.csv` — Metadata and splits
- `best_model.pt` — Saved model weights
- `download.sh` — Dataset download script

## Main Steps

1. **Data Preparation**
   - Group MRI slices into 3D volumes per subject
   - Save volumes as `.pt` files; store metadata in CSVs

2. **Dataset & DataLoader**
   - Custom PyTorch `Dataset` loads volumes and labels
   - Train/validation split via CSVs

3. **Model**
   - 3D CNN for binary classification (Alzheimer vs. Healthy)
   - Optionally use pretrained 3D ResNet

4. **Training & Experiment Tracking**
   - Weighted cross-entropy loss for class imbalance
   - Early stopping, LR scheduling, best model saving
   - **MLflow** is used to log parameters, metrics, model artifacts, and training curves for each experiment. You can view and compare runs in the MLflow UI.

5. **Evaluation**
   - Classification report, confusion matrix, loss curves
   - Metrics and artifacts are logged to MLflow

6. **Visualization**
   - Interactive 3D volume inspection with napari (`visualize.py`)

## Results
<img src=".github/training_plot.png" alt="training plot" width="400"/>
<img src=".github/confusion_matrix.png" alt="confusion matrix" width="400"/>



| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| **Alzheimer**| 0.62      | 0.89   | 0.73     | 63      |
| **Healthy**  | 0.96      | 0.83   | 0.89     | 206     |
|              |           |        |          |         |
| **Accuracy** |           |        | **0.84** | 269     |
| **Macro avg**| 0.79      | 0.86   | 0.81     | 269     |
| **Weighted avg** | 0.88  | 0.84   | 0.85     | 269     |

## Requirements

Install dependencies with:

```bash
pip install torch torchvision scikit-learn pandas matplotlib tqdm pillow napari mlflow
```

## Usage

1. **Download Data**
   - Download the dataset using the bash script

1. **Preprocess Data**
   - Run the notebook cells up to the preprocessing section to generate 3D volumes and metadata.

2. **Train Model**
   - Continue running the notebook to train the 3D CNN and monitor loss curves.

3. **Evaluate**
   - The notebook will output classification metrics and confusion matrix.

4. **Visualize Volumes**
   - Run `python visualize.py` to open a napari viewer for a sample MRI volume.