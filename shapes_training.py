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
dataset_path = './data/shapes_dataset_con'
kaggle.api.dataset_download_files(
    'mawins/shapes-sss-synthetic-img',
    path='./data',
    unzip=True
)

print("Download complete!")

# 2. Fix data.yaml paths
data_yaml_path = os.path.join(dataset_path, "data.yaml")
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Set the absolute path to the dataset folder
    data['path'] = os.path.abspath(os.path.join(dataset_path, 'dataset'))
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f)
    print(f"Updated {data_yaml_path} with path: {data['path']}")

# 3. Train YOLOv11
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
    plots=True
)
