import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import contextlib
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules import *
from ultralytics.utils.ops import make_divisible
import ultralytics.nn.tasks as tasks

# --- Re-apply the same custom modules and patches from BES_yolo.py ---
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

def custom_parse_model(d, ch, verbose=True):
    # This is a direct copy-paste of the patch in BES_yolo.py
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

def main():
    model_path = "best_scr.pt"
    data_yaml = "data/Combined_Dataset/data.yaml"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run validation
    print("Running validation...")
    results = model.val(data=data_yaml, project="investigation", name="val_results")
    
    # Extract metrics
    print("\n[ VALIDATION SUMMARY ]")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # Save a small sample of images where the model fails (confidence threshold is low)
    print("\nGenerating visual failure samples...")
    # These will be in investigation/val_results/
    # Ultralytics val() already saves val_batch0_labels.jpg, val_batch0_pred.jpg etc.
    
    print("\nInvestigation complete. Check 'investigation/val_results' for plots and matrices.")

if __name__ == "__main__":
    main()
