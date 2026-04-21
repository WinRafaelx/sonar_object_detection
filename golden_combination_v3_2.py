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

# --- 2. CUSTOM MODULES ---

class SonarSPDConv(nn.Module):
    def __init__(self, c1, c2, n=1, k=1, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2) if (k > 0 or c2 != c1 * 4) else nn.Identity()
    def forward(self, x):
        x1 = x[..., ::2, ::2]; x2 = x[..., 1::2, ::2]; x3 = x[..., ::2, 1::2]; x4 = x[..., 1::2, 1::2]
        return self.conv(torch.cat([x1, x2, x3, x4], dim=1))

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, n=1, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()

class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid() 
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.view(b * self.groups, -1, h, w) 
        x1 = self.conv1x1(nn.AdaptiveAvgPool2d(1)(group_x))
        x2 = self.conv3x3(group_x)
        out = self.sig(x1 * x2)
        return (out * group_x).view(b, c, h, w)

class BiFPN_Concat2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Concat2, self).__init__()
        self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.projections = nn.ModuleList([
            nn.Conv2d(ch_in, c2, kernel_size=1, stride=1, padding=0) if ch_in != c2 else nn.Identity()
            for ch_in in c1
        ])
    def forward(self, x):
        w = F.relu(self.w)
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        x_proj = [proj(feat) for proj, feat in zip(self.projections, x)]
        return (w.view(-1, 1, 1, 1, 1) * torch.stack(x_proj)).sum(dim=0)

# --- 3. MONKEY-PATCHING ---
def patch_yolo_parser():
    setattr(tasks, 'SonarSPDConv', SonarSPDConv)
    setattr(tasks, 'CoordAtt', CoordAtt)
    setattr(tasks, 'EMA', EMA)
    setattr(tasks, 'BiFPN_Concat2', BiFPN_Concat2)
    try:
        source = inspect.getsource(tasks.parse_model)
        if "SonarSPDConv" not in source:
            concat_branch = "elif m is Concat:\n            c2 = sum(ch[x] for x in f)"
            if concat_branch in source:
                bifpn_branch = concat_branch + "\n        elif m is BiFPN_Concat2:\n            c2 = args[0]\n            if c2 != nc:\n                c2 = make_divisible(min(c2, max_channels) * width, 8)\n            args = [[ch[x] for x in f], c2]"
                source = source.replace(concat_branch, bifpn_branch)

            else_branch = "else:\n            c2 = ch[f]"
            custom_branch = """elif m is SonarSPDConv:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {CoordAtt, EMA}:
            c2 = ch[f]
            args = [c2, *args]
        """ + else_branch
            
            if else_branch in source:
                source = source.replace(else_branch, custom_branch)

            exec_globals = tasks.__dict__.copy()
            exec_globals.update({
                'SonarSPDConv': SonarSPDConv, 'CoordAtt': CoordAtt, 
                'EMA': EMA, 'BiFPN_Concat2': BiFPN_Concat2, 'torch': torch, 'nn': nn, 
                'inspect': inspect, 'ast': __import__('ast'), 'contextlib': contextlib,
                'make_divisible': make_divisible
            })
            exec(source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser with explicit custom module logic.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser: {e}")

patch_yolo_parser()

# --- 4. CHAMPION ARCHITECTURE V3.2 (3-Scale simplified) ---
CHAMPION_YAML = """
scales:
  m: [0.50, 1.00, 512] 
backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, SonarSPDConv, [256]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]        # 2
  - [-1, 1, SonarSPDConv, [512]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]        # 4
  - [-1, 1, EMA, [32]]                 # 5-Attention on P3
  - [-1, 1, Conv, [512, 3, 2]]         # 6-P4/16
  - [-1, 2, C3k2, [512, True]]         # 7
  - [-1, 1, Conv, [1024, 3, 2]]        # 8-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 9
  - [-1, 1, SPPF, [1024, 5]]           # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 11 (P4)
  - [[-1, 7], 1, BiFPN_Concat2, [512]]  # 12-BiFPN Fusion
  - [-1, 2, C3k2, [512, False]]        # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 14 (P3)
  - [[-1, 5], 1, BiFPN_Concat2, [256]]  # 15-BiFPN Fusion
  - [-1, 2, C3k2, [256, False]]        # 16

  - [-1, 1, CoordAtt, [256, 256]]      # 17
  
  - [13, 1, Conv, [512, 3, 2]]         # 18
  - [[-1, 10], 1, BiFPN_Concat2, [512]] # 19
  - [-1, 2, C3k2, [512, True]]         # 20
  
  # Detect at 3 scales: P3, P4, P5
  - [[16, 13, 20], 1, Detect, [nc]]    # 21
"""

# --- 5. EXECUTION ---
def run_marathon(args):
    print("="*60)
    print("SONAR OBJECT DETECTION: GOLDEN COMBINATION V3.2 (3-SCALE)")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}")
    print("="*60)
    
    # 1. Dataset Setup
    base_data_dir = os.path.abspath('./data')
    sss_dir = os.path.join(base_data_dir, 'combined_data')
    sss_yaml_path = os.path.join(sss_dir, 'data.yaml')
    
    with open(sss_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    nc = data_config['nc']
    
    # 2. Hyperparams
    with open('best_hyperparameters.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    # 3. Build Champion Model
    model_yaml = "tmp_champion_v3_2.yaml"
    with open(model_yaml, "w") as f:
        f.write(f"nc: {nc}\n" + CHAMPION_YAML)
    
    model = YOLO(model_yaml).load("yolo11m.pt")
    
    # 4. Training
    project_dir = os.path.abspath("runs/champion_v3_2")
    
    print(f"\n>>> STARTING TRAINING V3.2...")
    model.train(
        data=sss_yaml_path, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        freeze=0, project=project_dir, name="final_champion", exist_ok=True, 
        patience=100, **hyp
    )
    
    print("\n" + "!"*60)
    print("V3.2 TRAINING COMPLETE!")
    print(f"Results saved to: {project_dir}/final_champion")
    print("!"*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    
    args = parser.parse_args()
    run_marathon(args)
