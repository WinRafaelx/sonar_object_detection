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
dataset_path = './data/Combined_Dataset'
kaggle.api.dataset_download_files(
    'paweekorns/sss-images',
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

model = YOLO("yolo11m.pt")

results = model.train(
    data=data_yaml_path,
    imgsz=640,
    epochs=500,
    patience=50,
    batch=batch_size,
    project="runs/train",
    name="yolo11m_sonar", 
    verbose=True,
    plots=True,

    degrees=10.0,      # Rotate image +/- 10 degrees randomly
    fliplr=0.5,        # Flip horizontal 50% of the time
    mixup=0.1,         # Mixup (blend 2 images) 10% chance
    scale=0.5,         # Zoom in/out by +/- 50%

    # Turn off unrelated augs
    flipud=0.0,        # Don't flip upside down (violates shadow physics)
    hsv_h=0.0,         # Don't change Hue (it's grayscale anyway)
    hsv_s=0.0,         # Don't change Saturation
    hsv_v=0.4          # Random brightness (Value) changes are okay
)
