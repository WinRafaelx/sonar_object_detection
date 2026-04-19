import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import inspect
import pandas as pd
from dotenv import load_dotenv
import argparse
import contextlib
import shutil
import time

# Ultralytics imports
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.utils.ops import make_divisible
from ultralytics.utils import LOGGER, colorstr

# 1. LOAD ENV & KAGGLE AUTH
load_dotenv()
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']
import kaggle

# --- 2. CUSTOM CHAMPION MODULES ---

class SonarSPDConv(nn.Module):
    def __init__(self, c1, c2, n=1, k=1, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2) if (k > 0 or c2 != c1 * 4) else nn.Identity()
    def forward(self, x):
        x1 = x[..., ::2, ::2]; x2 = x[..., 1::2, ::2]; x3 = x[..., ::2, 1::2]; x4 = x[..., 1::2, 1::2]
        return self.conv(torch.cat([x1, x2, x3, x4], dim=1))

class CBAM(nn.Module):
    def __init__(self, c1, c2=None, reduction=16, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c2 = c2 or c1
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c1, max(c1 // reduction, 1), 1, bias=False),
                                nn.ReLU(inplace=True), nn.Conv2d(max(c1 // reduction, 1), c1, 1, bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
        self.bn = nn.BatchNorm2d(c2) if c1 != c2 else nn.Identity()
    def forward(self, x):
        x = x * self.ca(x)
        res = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        x = x * self.sa(res)
        return self.bn(self.proj(x))

# --- 3. MONKEY-PATCHING ---
def patch_yolo_parser():
    setattr(tasks, 'SonarSPDConv', SonarSPDConv)
    setattr(tasks, 'CBAM', CBAM)
    try:
        source = inspect.getsource(tasks.parse_model)
        if "SonarSPDConv" not in source:
            for anchor in ["SPDConv,", "C2fCIB,", "C2f,", "Conv,"]:
                if anchor in source:
                    source = source.replace(anchor, f"{anchor} SonarSPDConv, CBAM,", 2)
                    break
            exec_globals = tasks.__dict__.copy()
            exec_globals.update({'SonarSPDConv': SonarSPDConv, 'CBAM': CBAM, 'torch': torch, 'nn': nn, 
                                 'inspect': inspect, 'ast': __import__('ast'), 'contextlib': contextlib})
            exec(source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser: {e}")

patch_yolo_parser()

# --- 4. CHAMPION ARCHITECTURE ---
CHAMPION_YAML = """
scales:
  m: [0.50, 1.00, 512] 
backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, SonarSPDConv, [256]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]        # 2
  - [-1, 1, SonarSPDConv, [512]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]        # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 10
  - [[-1, 6], 1, Concat, [1]]  # 11
  - [-1, 2, C3k2, [512, False]] # 12
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 13
  - [[-1, 4], 1, Concat, [1]]  # 14
  - [-1, 2, C3k2, [256, False]] # 15
  - [-1, 1, Conv, [256, 3, 2]] # 16
  - [[-1, 12], 1, Concat, [1]] # 17
  - [-1, 2, C3k2, [512, False]] # 18
  - [-1, 1, Conv, [512, 3, 2]] # 19
  - [[-1, 9], 1, Concat, [1]]  # 20
  - [-1, 2, C3k2, [1024, True]] # 21
  - [[15, 18, 21], 1, Detect, [nc]] # 22
"""

# --- 5. EXECUTION ---
def main():
    print("="*60)
    print("SONAR OBJECT DETECTION: GOLDEN COMBINATION MARATHON")
    print("="*60)
    
    # 1. Dataset Setup
    base_data_dir = os.path.abspath('./data')
    # This dataset unzips into 'Combined_Dataset'
    sss_dir = os.path.join(base_data_dir, 'combined_data')
    
    if not os.path.exists(sss_dir):
        dataset_name = 'mawins/sonar-image'
        print(f"Downloading new dataset: {dataset_name}...")
        kaggle.api.dataset_download_files(dataset_name, path=base_data_dir, unzip=True)
    
    sss_yaml_path = os.path.join(sss_dir, 'data.yaml')
    with open(sss_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    data_config['path'] = os.path.abspath(sss_dir)
    with open(sss_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    nc = data_config['nc']
    
    # 2. Hyperparams
    with open('best_hyperparameters.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    hyp['box'] = 10.0  # Recall-optimized gain
    
    # 3. Build Champion Model
    model_yaml = "tmp_champion.yaml"
    with open(model_yaml, "w") as f:
        f.write(f"nc: {nc}\n" + CHAMPION_YAML)
    
    model = YOLO(model_yaml).load("yolo11m.pt")
    
    # 4. Two-Stage Marathon
    project_dir = os.path.abspath("runs/champion")
    
    print("\n>>> STAGE 1: BACKBONE STABILIZATION (50 Epochs, Frozen)...")
    model.train(
        data=sss_yaml_path, imgsz=640, epochs=50, batch=16,
        freeze=10, project=project_dir, name="stage1", exist_ok=True, **hyp
    )
    
    print("\n>>> STAGE 2: THE 450-EPOCH HYPER-MARATHON (Unfrozen)...")
    best_weights = os.path.join(project_dir, "stage1", "weights", "best.pt")
    model = YOLO(best_weights)
    model.train(
        data=sss_yaml_path, imgsz=640, epochs=450, batch=16,
        freeze=0, project=project_dir, name="final_champion", exist_ok=True, **hyp
    )
    
    print("\n" + "!"*60)
    print("CHAMPION TRAINING COMPLETE!")
    print(f"Results saved to: {project_dir}/final_champion")
    print("!"*60)

if __name__ == "__main__":
    main()
