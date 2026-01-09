import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from roboflow import Roboflow

# --- 1. GLOBAL REGISTRATION ---
# These MUST be defined at the top level so they appear in globals()

class SPDConv(nn.Module):
    """ Space-to-Depth Convolution: Replaces strided convs to prevent 
    information loss in small targets. [cite: 33, 221] """
    def __init__(self, inc, ouc, k=1):
        super().__init__()
        self.conv = nn.Conv2d(inc * 4, ouc, k, padding=k//2)

    def forward(self, x):
        # Slice and stack sub-regions: Eq. 5 in paper [cite: 386]
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        # Concat in channel dimension: Eq. 6 in paper [cite: 389]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    """ Convolutional Block Attention Module: Dual attention (Spatial + Channel) 
    to focus on salient regions. [cite: 34, 163] """
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
        x = x * self.ca(x) # Channel Attention [cite: 404]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return x * self.sa(res) # Spatial Attention [cite: 405]

# --- 2. YAML ARCHITECTURE ---
SOCA_YOLO11M_YAML = """
# SOCA-YOLOv11 Unified Architecture for 4 Classes
nc: 4  # Correct: 4 types of targets
backbone:
  - [-1, 1, Conv, [64, 3, 2]]      # 0 - P1/2
  - [-1, 1, SPDConv, [128]]         # 1 - P2/4
  - [-1, 3, C3k2, [128, False]]    # 2
  - [-1, 1, SPDConv, [256]]         # 3 - P3/8
  - [-1, 6, C3k2, [256, False]]    # 4 (Target for Neck concat)
  - [-1, 1, CBAM, [256]]            # 5 - Backbone Attention
  - [-1, 1, Conv, [512, 3, 2]]      # 6 - P4/16 (Target for Neck concat)
  - [-1, 6, C3k2, [512, False]]    # 7
  - [-1, 1, Conv, [1024, 3, 2]]     # 8 - P5/32 (Target for Neck concat)
  - [-1, 3, C3k2, [1024, True]]    # 9
  - [-1, 1, SPPF, [1024, 5]]        # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 11
  - [[-1, 7], 1, Concat, [1]]      # 12 - cat backbone P4
  - [-1, 3, C3k2, [512, False]]    # 13
  - [-1, 1, CBAM, [512]]            # 14 - Neck Attention

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 15
  - [[-1, 4], 1, Concat, [1]]      # 16 - cat backbone P3
  - [-1, 3, C3k2, [256, False]]    # 17 (P3 Output)

  - [-1, 1, Conv, [256, 3, 2]]      # 18
  - [[-1, 14], 1, Concat, [1]]     # 19 - cat head P4
  - [-1, 3, C3k2, [512, False]]    # 20 (P4 Output)

  - [-1, 1, Conv, [512, 3, 2]]      # 21
  - [[-1, 10], 1, Concat, [1]]     # 22 - cat head P5
  - [-1, 3, C3k2, [1024, True]]    # 23 (P5 Output)

  - [[17, 20, 23], 1, Detect, [nc]] # Final Output Layer
"""

def main():
    # Save the architecture
    with open("soca_yolo11m.yaml", "w") as f:
        f.write(SOCA_YOLO11M_YAML)

    # Roboflow Setup
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    version = project.version(1)
    dataset = version.download("yolov11")
    data_yaml = os.path.join(dataset.location, "data.yaml")

    # The YOLO constructor will now find SPDConv and CBAM in globals()
    model = YOLO("soca_yolo11m.yaml")

    # Train on A40
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=32,
        device=0,
        project="runs/train",
        name="yolo11m_soca_fixed"
    )

if __name__ == '__main__':
    main()