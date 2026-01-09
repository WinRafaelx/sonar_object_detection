import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C3k2
from roboflow import Roboflow

# --- 1. RESEARCH-BACKED MODULES ---

class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution: Replaces strided convs to prevent 
    information loss in small underwater targets. [cite: 33, 221]
    """
    def __init__(self, inc, ouc, k=1):
        super().__init__()
        self.conv = nn.Conv2d(inc * 4, ouc, k, padding=k//2)

    def forward(self, x):
        # Slice and stack: Eq. 5 in paper [cite: 386]
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        # Concat in channel dimension: Eq. 6 in paper [cite: 389]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: Dual attention (Spatial + Channel)
    to suppress underwater noise. [cite: 34, 163]
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention [cite: 404]
        x = x * self.ca(x)
        # Spatial Attention [cite: 405]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return x * self.sa(res)

# --- 2. MODEL DEFINITION (YAML STRING) ---

# We replace standard Conv(stride=2) with SPDConv to preserve features [cite: 93, 268]
# and inject CBAM into the Backbone and Head[cite: 395].
SOCA_YOLO11M_YAML = """
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # P1/2
  - [-1, 1, SPDConv, [128]]     # P2/4 (Replaces standard stride-2)
  - [-1, 3, C3k2, [128, False]]
  - [-1, 1, SPDConv, [256]]     # P3/8
  - [-1, 6, C3k2, [256, False]]
  - [-1, 1, CBAM, [256]]        # Backbone Attention [cite: 395]
  - [-1, 1, Conv, [512, 3, 2]]  # P4/16
  - [-1, 6, C3k2, [512, False]]
  - [-1, 1, Conv, [1024, 3, 2]] # P5/32
  - [-1, 3, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C3k2, [512, False]]
  - [-1, 1, CBAM, [512]]        # Neck Attention [cite: 415]

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C3k2, [256, False]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C3k2, [512, False]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C3k2, [1024, True]]

  - [[16, 19, 22], 1, Detect, [nc]] # Detect (P3, P4, P5)
"""

# --- 3. MAIN TRAINING FLOW ---

def main():
    # Write custom YAML to disk
    with open("soca_yolo11m.yaml", "w") as f:
        f.write(SOCA_YOLO11M_YAML)

    # Roboflow Setup
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    version = project.version(1)
    dataset = version.download("yolov11")
    data_yaml = os.path.join(dataset.location, "data.yaml")

    # Initialize custom model
    # Note: Since architecture changed, we start from scratch or map weights
    model = YOLO("soca_yolo11m.yaml")

    # Train on NVIDIA A40
    # Higher batch size and imgsz 640 as per paper [cite: 36, 426]
    results = model.train(
        data=data_yaml,
        imgsz=640,          # Paper uses 640x640 [cite: 426]
        epochs=100,         # Increased epochs for convergence
        batch=32,           # A40 has 48GB VRAM, push this high!
        device=0,           # Force A40 usage
        project="runs/train",
        name="yolo11m_soca_unified",
        optimizer='SGD',    # Paper likely uses SGD for stability [cite: 617]
        patience=20,
        verbose=True
    )

if __name__ == '__main__':
    main()