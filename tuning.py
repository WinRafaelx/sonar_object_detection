from dotenv import load_dotenv
import os
import yaml
from ultralytics import YOLO
from constant import batch_size

# 1. Load variables from .env file BEFORE importing kaggle
load_dotenv()

# Map KAGGLE_API_TOKEN to KAGGLE_KEY for the kaggle library
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']

# Now import kaggle - the library calls authenticate() on import
try:
    import kaggle
    # Download your dataset
    dataset_path = './data/sampled_yolo_dataset'
    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(
            'mawins/sample-sss-img',
            path='./data',
            unzip=True
        )
        print("Download complete!")
    else:
        print("Dataset already exists, skipping download.")
except ImportError:
    print("Kaggle library not found. Ensure dataset is available at ./data/sampled_yolo_dataset")
    dataset_path = './data/sampled_yolo_dataset'

data_yaml_path = os.path.join(dataset_path, "data.yaml")

# --- FIX: Patch the data.yaml with absolute paths ---
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Tell YOLO exactly where the root folder is
    yaml_data['path'] = os.path.abspath(dataset_path)

    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
else:
    print(f"Warning: {data_yaml_path} not found.")

# 2. Load the YOLO11m model
model = YOLO("yolo11n.pt")

# 3. Start the automated experiment (Hyperparameter Tuning)
# Note: model.tune() will run multiple training iterations to find the best hyperparameters.
# It uses Genetic Algorithm (GA) by default to evolve the hyperparameters.
model.tune(
    data=data_yaml_path, 
    epochs=30,        # Training epochs per trial (30 is a good balance for tuning)
    iterations=100,   # Number of different parameter sets to try
    optimizer="AdamW", 
    plots=False,      # Set to True to see evolution plots
    save=False, 
    val=False,
    imgsz=640         # Consistent with your training resolution
)
