import os
import sys
import torch
import torch.nn as nn
import yaml
import inspect
import contextlib
from dotenv import load_dotenv
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# Import custom modules
from custom_modules import SonarSPDConv, CoordAtt, CBAM, EMA, BiFPN_Concat2
from constant import batch_size as default_batch_size

# 1. LOAD ENV & KAGGLE AUTH
load_dotenv()
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']
import kaggle

# --- 2. MONKEY-PATCHING ---
def patch_yolo_parser():
    """Register custom modules and patch the YOLO model parser."""
    setattr(tasks, 'SonarSPDConv', SonarSPDConv)
    setattr(tasks, 'CoordAtt', CoordAtt)
    setattr(tasks, 'CBAM', CBAM)
    setattr(tasks, 'EMA', EMA)
    setattr(tasks, 'BiFPN_Concat2', BiFPN_Concat2)
    
    try:
        source = inspect.getsource(tasks.parse_model)
        if "SonarSPDConv" not in source:
            patched = False
            for anchor in ["SPDConv,", "C2fCIB,", "C2f,", "Conv,"]:
                if anchor in source:
                    source = source.replace(anchor, f"{anchor} SonarSPDConv, CoordAtt, CBAM,", 2)
                    patched = True
                    break
            
            concat_branch = "elif m is Concat:\n            c2 = sum(ch[x] for x in f)"
            if concat_branch in source:
                bifpn_branch = concat_branch + "\n        elif m is BiFPN_Concat2:\n            c2 = args[0]\n            if c2 != nc:\n                c2 = make_divisible(min(c2, max_channels) * width, 8)\n            args = [[ch[x] for x in f], c2]"
                source = source.replace(concat_branch, bifpn_branch)
            
            ema_branch = "\n        elif m is EMA:\n            c2 = ch[f]\n            args = [c2, *args]"
            source = source.replace("else:\n            c2 = ch[f]", ema_branch + "\n        else:\n            c2 = ch[f]")

            exec_globals = tasks.__dict__.copy()
            exec_globals.update({
                'SonarSPDConv': SonarSPDConv,
                'CoordAtt': CoordAtt,
                'CBAM': CBAM,
                'EMA': EMA,
                'BiFPN_Concat2': BiFPN_Concat2,
                'torch': torch,
                'nn': nn,
                'inspect': inspect,
                'ast': __import__('ast'),
                'contextlib': contextlib,
            })
            exec(source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser: {e}")

# Apply the patch immediately
patch_yolo_parser()

# --- 3. HELPERS ---
def patch_data_yaml(yaml_path, dataset_dir):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data['path'] = os.path.abspath(dataset_dir)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return os.path.abspath(yaml_path)

def train_sonar():
    # 1. Dataset Setup (Same as normal.py)
    base_data_dir = os.path.abspath('./data')
    # This dataset typically unzips into a folder
    sss_dir = os.path.join(base_data_dir, 'combined_data')
    
    if not os.path.exists(sss_dir):
        os.makedirs(base_data_dir, exist_ok=True)
        dataset_name = 'mawins/sonar-image'
        print(f"Downloading new dataset: {dataset_name}...")
        kaggle.api.dataset_download_files(dataset_name, path=base_data_dir, unzip=True)
        
        # Check for potential folder name variations after unzip
        downloaded_folders = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
        if 'Combined_Dataset' in downloaded_folders and not os.path.exists(sss_dir):
            os.rename(os.path.join(base_data_dir, 'Combined_Dataset'), sss_dir)
        elif 'sonar-image' in downloaded_folders and not os.path.exists(sss_dir):
            os.rename(os.path.join(base_data_dir, 'sonar-image'), sss_dir)
        elif not os.path.exists(sss_dir) and downloaded_folders:
            # If there's only one folder and it's not what we expect, maybe use it?
            # But let's stick to the folder containing data.yaml
            for folder in downloaded_folders:
                if os.path.exists(os.path.join(base_data_dir, folder, 'data.yaml')):
                    os.rename(os.path.join(base_data_dir, folder), sss_dir)
                    break

    data_yaml_path = os.path.join(sss_dir, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Could not find data.yaml in {sss_dir}")

    sss_yaml = patch_data_yaml(data_yaml_path, sss_dir)

    # 2. Initialize the model from your custom YAML structure
    # Note: Using the YAML requires nc to match the dataset if it's explicitly defined there.
    # YOLO usually overrides this during training.
    model_yaml = os.path.join(os.path.dirname(__file__), "yolo11-sonar.yaml")
    model = YOLO(model_yaml)

    # 3. Load weights from standard YOLO11
    model.load("yolo11m.pt") 

    # 4. Start Training
    model.train(
        data=sss_yaml,
        imgsz=640,
        epochs=500,
        batch=default_batch_size,
        project="runs/train",
        name="yolo11m_sonar", 
        verbose=True,
        plots=True 
    )

if __name__ == "__main__":
    train_sonar()
