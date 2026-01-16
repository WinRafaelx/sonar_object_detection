#!./venv/Scripts/python
import os
import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import ast
import contextlib
from copy import deepcopy
from glob import glob
from roboflow import Roboflow

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.nn.modules import *  # Import all modules
from ultralytics.utils.ops import make_divisible
from ultralytics.utils import LOGGER, colorstr
import ultralytics.nn.tasks as tasks

# Fix: Ensure Upsample is available for custom_parse_model if referenced as 'Upsample' in YAML
Upsample = nn.Upsample

# --- 1. Paper Preprocessing: 2D Discrete Wavelet Transform ---
def apply_wavelet_preprocessing(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    channels = cv2.split(img)
    processed_channels = []
    
    for ch in channels:
        coeffs = pywt.dwt2(ch, 'haar')
        LL, (LH, HL, HH) = coeffs
        # Normalize LL to 0-255 range instead of reconstructing
        LL_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
        processed_channels.append(np.uint8(LL_norm))
    
    # LL is half size, so we resize back to original for YOLO compatibility
    final = cv2.merge(processed_channels)
    final = cv2.resize(final, (img.shape[1], img.shape[0]))
    cv2.imwrite(image_path, final)

# --- 2. Custom Modules: EMA & BiFPN ---
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid() # Use Sigmoid for attention masks

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.view(b * self.groups, -1, h, w) 
        x1 = nn.AdaptiveAvgPool2d(1)(group_x)
        x1 = self.conv1x1(x1)
        x2 = self.conv3x3(group_x)
        # Apply Sigmoid mask
        out = self.sig(x1 * x2)
        return (out * group_x).view(b, c, h, w)

class BiFPN_Concat2(nn.Module):
    """ Bi-directional Feature Pyramid Network fusion """
    def __init__(self, c1, c2):
        super(BiFPN_Concat2, self).__init__()
        # c1 is list of input channels, c2 is output channels
        self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        
        self.projections = nn.ModuleList()
        for ch_in in c1:
            if ch_in != c2:
                self.projections.append(nn.Conv2d(ch_in, c2, kernel_size=1, stride=1, padding=0))
            else:
                self.projections.append(nn.Identity())

    def forward(self, x):
        w = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        x_proj = [self.projections[i](feat) for i, feat in enumerate(x)]
        
        # Weighted sum fusion
        res = torch.zeros_like(x_proj[0])
        for i, feat in enumerate(x_proj):
            res += w[i] * feat
        return res

# --- 3. Monkey-patch parse_model ---

def custom_parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    
    # Args
    legacy = True
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    
    # Modules definitions (copied from ultralytics source logic)
    base_modules = frozenset({Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f})
    repeat_modules = frozenset({BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f})

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        
        # Custom Module Logic
        if m is BiFPN_Concat2:
            c2 = args[0]
            c1 = [ch[x] for x in f]
            args = [c1, c2, *args[1:]]
        # Standard Logic
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
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)

# Patch the tasks module
tasks.parse_model = custom_parse_model
# Ensure EMA and Upsample are found
tasks.EMA = EMA
tasks.Upsample = nn.Upsample

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

  - [[15, 18, 21], 1, Detect, [nc]] 
"""

def main():
    # 1. Dataset Setup
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    dataset = project.version(1).download("yolov11")

    # 2. Wavelet Preprocessing
    print("Applying 2D-DWT Preprocessing...")
    for img_path in glob(os.path.join(dataset.location, "**/*.jpg"), recursive=True):
        apply_wavelet_preprocessing(img_path)

    # 3. Save YAML
    import yaml
    with open(os.path.join(dataset.location, "data.yaml"), "r") as f:
        data_config = yaml.safe_load(f)
    nc = data_config["nc"]

    yaml_path = "bes_yolo_config.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"nc: {nc}\n" + BES_YOLO_YAML)

    # 4. Training
    model = YOLO(yaml_path) 
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        imgsz=640,
        epochs=500,
        patience=50,
        batch=16,            
        optimizer='SGD',     
        lr0=0.01,            
        momentum=0.937,      
        warmup_epochs=3,     
        project="runs/train",
        name="bes_yolo_11_implementation"
    )

if __name__ == "__main__":
    main()
