#!./venv/Scripts/python
import os
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
        # Read as Grayscale (1-channel). Sonar isn't color! 3x less math instantly.
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
        print(f"Skipping DWT, '{dest_dir}' already exists. Big brain caching.")
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
        
        # Max performant pure matrix math. Reshape w for 5D stack [N, B, C, H, W]
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
    # 1. Load environment variables for Kaggle API
    from dotenv import load_dotenv
    load_dotenv()
    if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
        os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']

    # 2. Download Kaggle Dataset (Match Let_start_train.py)
    dataset_root = './data/Combined_Dataset'
    import kaggle
    kaggle.api.dataset_download_files(
        'paweekorns/sss-images',
        path='./data',
        unzip=True
    )
    
    # 3. Apply DWT Preprocessing (Maintain BES logic)
    dwt_dataset_loc = setup_dwt_dataset(os.path.abspath(dataset_root))

    # 3. Configure Model YAML
    yaml_path = os.path.join(dwt_dataset_loc, "bes_yolo_config.yaml")
    with open(os.path.join(dwt_dataset_loc, "data.yaml"), "r") as f:
        config = yaml.safe_load(f)
        nc = config["nc"]

    with open(yaml_path, "w") as f:
        f.write(f"nc: {nc}\n" + BES_YOLO_YAML)

    # 4. Train
    model = YOLO(yaml_path) 
    model.train(
        data=os.path.join(dwt_dataset_loc, "data.yaml"),
        imgsz=640,
        epochs=500,
        patience=50,
        batch=16,            
        optimizer='AdamW',   # Better for custom attention/BiFPN modules
        lr0=0.001,          # Standard starting LR for AdamW
        cos_lr=True,        # Use cosine learning rate scheduler
        warmup_epochs=3,     
        
        # Sonar-optimized augmentations (Respecting shadow physics)
        degrees=10.0,        # Small rotations
        fliplr=0.5,          # Horizontal flip is fine
        flipud=0.0,          # Vertical flip disabled (breaks sonar shadow logic)
        mixup=0.1,           # Helps with noise
        scale=0.5,           # Multi-scale robust
        hsv_h=0.0,           # No hue changes (grayscale)
        hsv_s=0.0,           # No saturation changes
        hsv_v=0.4,           # Brightness variations are common in sonar
        
        project="runs/train",
        name="bes_yolo_11_refined"
    )

if __name__ == "__main__":
    main()