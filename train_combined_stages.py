import os
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import inspect
import yaml
from dotenv import load_dotenv

# 1. LOAD ENV & KAGGLE AUTH
load_dotenv()
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']
import kaggle

from constant import batch_size

# --- 2. RESEARCH & REGISTER MODULES (Same as train.py) ---
try:
    from .custom_modules import SonarSPDConv, CoordAtt
except (ImportError, ValueError):
    from custom_modules import SonarSPDConv, CoordAtt

setattr(tasks, 'SonarSPDConv', SonarSPDConv)
setattr(tasks, 'CoordAtt', CoordAtt)

def patch_parse_model():
    try:
        source = inspect.getsource(tasks.parse_model)
        if "SonarSPDConv" not in source:
            # Fix: Only replace the first occurrence to avoid inject in repeat_modules
            new_source = source.replace("A2C2f,", "A2C2f, SonarSPDConv, CoordAtt,", 1)
            exec_globals = tasks.__dict__.copy()
            exec_globals.update({
                'SonarSPDConv': SonarSPDConv,
                'CoordAtt': CoordAtt,
                'torch': torch,
                'nn': nn,
                'inspect': inspect,
            })
            exec(new_source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser.")
        else:
            print("✓ YOLO model parser already patched.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser: {e}")

patch_parse_model()

def patch_data_yaml(yaml_path, dataset_dir):
    """Updates the 'path' in data.yaml and ensures train/val/test are relative."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Set the root path for the dataset
    data['path'] = os.path.abspath(dataset_dir)
    
    # Ensure train/val/test are relative to that path
    if 'train' in data and isinstance(data['train'], str):
        if 'train' in data['train']: # If it was an absolute path containing 'train'
             data['train'] = 'train/images'
    if 'val' in data and isinstance(data['val'], str):
        data['val'] = 'test/images' # URPC2020 often uses 'test' as 'val'
    if 'test' in data and isinstance(data['test'], str):
        data['test'] = 'test/images'
        
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return os.path.abspath(yaml_path)

def push_model_to_git(file_path):
    """Adds, commits, and pushes a specific file to Git."""
    import subprocess
    
    try:
        if not os.path.exists(file_path):
            print(f"Error: Could not find model file at {file_path}")
            return

        print(f"\nStaging {file_path}...")
        subprocess.run(["git", "add", "-f", file_path], check=True)
        
        # Also stage .gitignore if it changed
        subprocess.run(["git", "add", ".gitignore"], check=True)
        
        commit_msg = f"Auto-commit: New trained sonar model weights ({os.path.basename(file_path)})"
        print(f"Committing changes: '{commit_msg}'...")
        subprocess.run(["git", "commit", "-m", commit_msg, "--allow-empty"], check=True)
        
        print("Pushing to remote repository...")
        subprocess.run(["git", "push"], check=True)
        print("✓ Successfully pushed model to Git!")

    except subprocess.CalledProcessError as e:
        print(f"⚠ Git operation failed: {e}")
    except Exception as e:
        print(f"⚠ Unexpected error during Git push: {e}")

def main():
    # Directories
    base_data_dir = os.path.abspath('./data')
    os.makedirs(base_data_dir, exist_ok=True)

    # --- STAGE 1: PRE-TRAIN ON URPC2020 ---
    print("\n--- STAGE 1: PRE-TRAINING ON URPC2020 ---")
    urpc_dataset = 'lywang777/urpc2020'
    urpc_dir = os.path.join(base_data_dir, 'URPC2020')
    
    if not os.path.exists(urpc_dir):
        print(f"Downloading {urpc_dataset}...")
        kaggle.api.dataset_download_files(urpc_dataset, path=base_data_dir, unzip=True)
    
    urpc_yaml_path = patch_data_yaml(os.path.join(urpc_dir, 'data.yaml'), urpc_dir)

    # Initialize model with custom YAML
    model_yaml = os.path.join(os.path.dirname(__file__), "yolo11-sonar.yaml")
    model = YOLO(model_yaml)
    model.load("yolo11m.pt") 

    # Train Stage 1
    model.train(
        data=urpc_yaml_path,
        epochs=100, 
        imgsz=640,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="runs/train",
        name="stage1_urpc",
        optimizer='AdamW'
    )
    
    stage1_weights = os.path.abspath("runs/detect/runs/train/stage1_urpc/weights/best.pt")
    if not os.path.exists(stage1_weights):
        print("Error: Stage 1 training failed to produce weights.")
        return

    # --- STAGE 2: FINE-TUNE ON SSS-IMAGES ---
    print("\n--- STAGE 2: FINE-TUNING ON SSS-IMAGES ---")
    sss_dataset = 'paweekorns/sss-images'
    sss_dir = os.path.join(base_data_dir, 'SSS_merged')
    
    if not os.path.exists(sss_dir):
        print(f"Downloading {sss_dataset}...")
        kaggle.api.dataset_download_files(sss_dataset, path=base_data_dir, unzip=True)
    
    sss_yaml_path = patch_data_yaml(os.path.join(sss_dir, 'data.yaml'), sss_dir)

    # Load weights from Stage 1
    model_stage2 = YOLO(stage1_weights)

    # Train Stage 2
    model_stage2.train(
        data=sss_yaml_path,
        epochs=100,
        imgsz=640,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="runs/train",
        name="stage2_sss_finetune",
        optimizer='AdamW',
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        lr0=0.001
    )

    # --- STAGE 3: AUTO-PUSH TO GIT ---
    print("\n--- STAGE 3: AUTO-PUSHING MODEL TO GIT ---")
    sss_weights_path = os.path.abspath("runs/train/stage2_sss_finetune/weights/best.pt")
    push_model_to_git(sss_weights_path)

if __name__ == "__main__":
    main()
