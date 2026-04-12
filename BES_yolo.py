#!./venv/Scripts/python
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import contextlib
import shutil
import yaml
from glob import glob
from roboflow import Roboflow
from pytorch_wavelets import DWTForward
from torch.utils.data import Dataset, DataLoader

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.nn.modules import *
from ultralytics.utils.ops import make_divisible
from ultralytics.utils import LOGGER, colorstr
import ultralytics.nn.tasks as tasks
from constant import batch_size

Upsample = nn.Upsample

# --- 1. Preprocessing: Cached 2D-DWT (Batched GPU Accelerated) ---
class SonarDataset(Dataset):
    """Feeds the GPU images in batches to stop it from waiting on the hard drive."""
    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # Read as Grayscale (1-channel). Sonar isn't color!
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to uniform shape for batching
        img_resized = cv2.resize(img, (640, 640))
        tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        
        # Return original dims so we can resize back before saving
        return tensor, path, img.shape[0], img.shape[1]


def setup_dwt_dataset(src_dir):
    """Max performant GPU-accelerated DWT caching with DataLoader batching."""
    dest_dir = src_dir + "_dwt"
    if os.path.exists(dest_dir):
        print(f"✓ Skipping DWT, '{dest_dir}' already exists. Big brain caching.")
        return dest_dir

    print(f"Applying Batched GPU-accelerated 2D-DWT to {dest_dir}...")
    shutil.copytree(src_dir, dest_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    
    paths = glob(os.path.join(dest_dir, "**/*.jpg"), recursive=True)
    dataset = SonarDataset(paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for batch_tensors, batch_paths, heights, widths in dataloader:
        batch_tensors = batch_tensors.to(device)
        
        with torch.no_grad():
            Yl, _ = dwt(batch_tensors)
            LL = Yl.squeeze(1) 
            
            # Fast GPU normalization across batch
            LL_min = LL.amin(dim=(1, 2), keepdim=True)
            LL_max = LL.amax(dim=(1, 2), keepdim=True)
            LL_norm = (LL - LL_min) / (LL_max - LL_min + 1e-8) * 255.0
            
            # Resize on GPU
            final_tensors = F.interpolate(
                LL_norm.unsqueeze(1), 
                size=(640, 640), 
                mode='bilinear', 
                align_corners=False
            )
            # Back to CPU
            final_arrays = final_tensors.squeeze(1).cpu().numpy().astype(np.uint8)
            
        # Save back to disk
        for i, path in enumerate(batch_paths):
            h, w = heights[i].item(), widths[i].item()
            final_img = cv2.resize(final_arrays[i], (w, h))
            cv2.imwrite(path, final_img)

    yaml_path = os.path.join(dest_dir, "data.yaml")
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_config['path'] = os.path.abspath(dest_dir)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
        
    return dest_dir

# --- 2. Custom Modules: EMA & BiFPN ---
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

# --- 3. Monkey-patch parse_model ---
def custom_parse_model(d, ch, verbose=True):
    legacy = True
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale: scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]

    if act: Conv.default_act = eval(act)

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    
    base_modules = frozenset({Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f})
    repeat_modules = frozenset({BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f})

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = getattr(torch.nn, m[3:]) if "nn." in m else getattr(__import__("torchvision").ops, m[16:]) if "torchvision.ops." in m else globals()[m]
        
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = max(round(n * depth), 1) if n > 1 else n
        
        if m is BiFPN_Concat2:
            c2 = args[0]
            c1 = [ch[x] for x in f]
            args = [c1, c2, *args[1:]]
        elif m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
        elif m is AIFI: args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer: c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d: args = [ch[f]]
        elif m is Concat: c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder: args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2, c1 = args[0], ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse: c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2, c1 = args[0], ch[f]
            args = [*args[1:]]
        else: c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0: ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)

tasks.parse_model = custom_parse_model
tasks.EMA = EMA
tasks.Upsample = nn.Upsample
tasks.BiFPN_Concat2 = BiFPN_Concat2

# --- 4. YAML Configuration String ---
BES_YOLO_YAML = """
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, EMA, [1024]]       # EMA Attention
  - [-1, 1, SPPF, [1024, 5]]   # 10

head:
  - [-1, 1, Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, BiFPN_Concat2, [512]] # BiFPN Fusion
  - [-1, 3, C2f, [512]] 

  - [-1, 1, Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, BiFPN_Concat2, [256]] # BiFPN Fusion
  - [-1, 3, C2f, [256]] 

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, BiFPN_Concat2, [512]] # BiFPN Fusion
  - [-1, 3, C2f, [512]] 

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, BiFPN_Concat2, [1024]] # BiFPN Fusion
  - [-1, 3, C2f, [1024]] 

  - [[16, 19, 22], 1, Detect, [nc]] 
"""

def main():
    from dotenv import load_dotenv
    load_dotenv()
    if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
        os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']
    
    kaggle_creds_found = ('KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ) or os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))

    # Download Kaggle Dataset
    dataset_path = './data/Combined_Dataset'
    if not os.path.exists(os.path.join(dataset_path, "data.yaml")):
        if not kaggle_creds_found:
            print("⚠ ERROR: Kaggle credentials not found! Place 'kaggle.json' in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY.")
            sys.exit(1)
        
        import kaggle
        print("\nDownloading dataset from Kaggle...")
        try:
            kaggle.api.dataset_download_files(
                'paweekorns/sss-images',
                path='./data',
                unzip=True
            )
            print("✓ Download complete!")
        except Exception as e:
            print(f"⚠ Kaggle download failed: {e}")
            sys.exit(1)
    else:
        print("✓ Dataset found locally, skipping download.")

    data_yaml = os.path.join(dataset_path, "data.yaml")

    # Patch the data.yaml with absolute paths to ensure YOLO finds the images
    with open(data_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    nc = yaml_data.get("nc", 4) # Extract nc from dataset
    yaml_data['path'] = os.path.abspath(dataset_path)
    
    with open(data_yaml, 'w') as f:
        yaml.dump(yaml_data, f)
    
    # Configure Model YAML
    yaml_path = "bes_yolo_config.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"nc: {nc}\n" + BES_YOLO_YAML)

    # Shared training parameters
    train_args = {
        'data': data_yaml,
        'imgsz': 640,
        'epochs': 500,
        'batch': batch_size,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'cos_lr': True,
        'warmup_epochs': 3,
        'patience': 50,
        'degrees': 10.0,
        'fliplr': 0.5,
        'flipud': 0.0,
        'mixup': 0.1,
        'scale': 0.5,
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.4,
        'project': "runs/train",
    }

    # --- RUN 1: BARE FROM SCRATCH ---
    print("\n" + "="*60)
    print("RUN 1: TRAINING BES-YOLO FROM SCRATCH")
    print("="*60 + "\n")
    model_scratch = YOLO(yaml_path)
    results_scratch = model_scratch.train(name="bes_yolo_scratch", **train_args)

    # --- RUN 2: PRETRAINED MODEL ---
    print("\n" + "="*60)
    print("RUN 2: TRAINING BES-YOLO WITH PRETRAINED WEIGHTS")
    print("="*60 + "\n")
    # Using yolo11m.pt as the channels (64, 128...) match the Medium scale
    model_pretrained = YOLO(yaml_path).load("yolo11m.pt")
    results_pretrained = model_pretrained.train(name="bes_yolo_pretrained", **train_args)

    # --- FINAL COMPARISON ---
    print("\n" + "="*60)
    print("FINAL ACCURACY METRIC COMPARISON (BES-YOLO)")
    print("="*60)
    
    def get_metrics(results):
        names = results.names
        box = results.box
        
        per_class = {}
        for i, cls_idx in enumerate(box.ap_class_index):
            c_idx = int(cls_idx)
            class_name = names.get(c_idx, str(c_idx))
            per_class[class_name] = box.ap50[i]
            
        return {
            "overall": {
                "mAP50": box.map50,
                "mAP50-95": box.map,
                "Precision": box.mp,
                "Recall": box.mr
            },
            "per_class": per_class
        }

    m_scratch = get_metrics(results_scratch)
    m_pretrained = get_metrics(results_pretrained)

    print("\n[ OVERALL METRICS ]")
    header = f"{'Metric':<15} | {'Scratch':<15} | {'Pretrained':<15} | {'Diff':<15}"
    print(header)
    print("-" * len(header))
    for key in m_scratch["overall"].keys():
        s_val, p_val = m_scratch["overall"][key], m_pretrained["overall"][key]
        print(f"{key:<15} | {s_val:<15.4f} | {p_val:<15.4f} | {p_val - s_val:<+15.4f}")

    print("\n[ PER-CLASS mAP50 ]")
    class_header = f"{'Class Name':<15} | {'Scratch':<15} | {'Pretrained':<15} | {'Diff':<15}"
    print(class_header)
    print("-" * len(class_header))
    
    all_classes = set(m_scratch["per_class"].keys()).union(set(m_pretrained["per_class"].keys()))
    for cls_name in sorted(all_classes):
        s_val = m_scratch["per_class"].get(cls_name, 0.0)
        p_val = m_pretrained["per_class"].get(cls_name, 0.0)
        print(f"{cls_name:<15} | {s_val:<15.4f} | {p_val:<15.4f} | {p_val - s_val:<+15.4f}")

if __name__ == "__main__":
    main()
