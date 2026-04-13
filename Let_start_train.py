from dotenv import load_dotenv
import os

# 1. Load variables from .env file BEFORE importing kaggle
load_dotenv()

# Map KAGGLE_API_TOKEN to KAGGLE_KEY for the kaggle library
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']

# Now import kaggle - the library calls authenticate() on import
import kaggle
import yaml
from ultralytics import YOLO
from constant import batch_size

# Download your dataset
dataset_path = './data/sampled_yolo_dataset'
kaggle.api.dataset_download_files(
    'mawins/sample-sss-img',
    path='./data',
    unzip=True
)

print("Download complete!")

data_yaml_path = os.path.join(dataset_path, "data.yaml")

# --- FIX: Patch the data.yaml with absolute paths ---
with open(data_yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

# Tell YOLO exactly where the root folder is
yaml_data['path'] = os.path.abspath(dataset_path)

with open(data_yaml_path, 'w') as f:
    yaml.dump(yaml_data, f)

# --- Load best hyperparameters ---
with open('best_hyperparameters.yaml', 'r') as f:
    best_hyp = yaml.safe_load(f)

model = YOLO("yolo11m.pt")

results = model.train(
    data=data_yaml_path,
    imgsz=640,
    epochs=100,
    patience=50,
    batch=batch_size,
    project="runs/train",
    name="yolo11m_sonar", 
    verbose=True,
    plots=True,
    **best_hyp
)
