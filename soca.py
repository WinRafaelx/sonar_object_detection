import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from roboflow import Roboflow
from constant import batch_size

# --- 1. RESEARCH MODULES (FIXED FOR YOLO PARSER) ---
class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution: Replaces strided convs to prevent information loss in small targets.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Args:
            c1: input channels
            c2: output channels
            k: kernel size
            s: stride (not used, kept for compatibility)
            p: padding (not used, kept for compatibility)
            g: groups (not used, kept for compatibility)
            d: dilation (not used, kept for compatibility)
            act: activation (not used, kept for compatibility)
        """
        super().__init__()
        # Internal conv input is c1 * 4 because of the spatial-to-depth shift
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2)

    def forward(self, x):
        # Slice and stack sub-regions: H, W -> H/2, W/2 and C -> 4C
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: Dual attention (Spatial + Channel)
    to focus on salient regions in noisy sonar data.
    """
    def __init__(self, c1, c2=None, reduction=16, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Args:
            c1: input channels
            c2: output channels (if None, same as c1)
            reduction: channel reduction ratio for attention
            k, s, p, g, d, act: kept for YOLO parser compatibility
        """
        super().__init__()
        c2 = c2 or c1
        
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c1 // reduction, 1), c1, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # If c1 != c2, add projection layer
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # Channel attention
        x = x * self.ca(x)
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        x = x * self.sa(res)
        # Project if needed
        return self.proj(x)

# --- 2. GLOBAL REGISTRATION ---
# Injecting into the task registry so the YAML parser recognizes the names
setattr(tasks, 'SPDConv', SPDConv)
setattr(tasks, 'CBAM', CBAM)

# --- 3. ARCHITECTURE DEFINITION (SOCA-YOLO11M) ---
SOCA_YOLO11M_YAML = """
nc: 4  # Number of sonar object classes

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0 - P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1 - P2/4 (regular conv instead of SPDConv here)
  - [-1, 3, C3k2, [128, False]]     # 2
  - [-1, 1, Conv, [256, 3, 2]]      # 3 - P3/8 (regular conv)
  - [-1, 6, C3k2, [256, False]]     # 4
  - [-1, 1, CBAM, [256]]            # 5 - Backbone Attention
  - [-1, 1, Conv, [512, 3, 2]]      # 6 - P4/16
  - [-1, 6, C3k2, [512, False]]     # 7
  - [-1, 1, Conv, [1024, 3, 2]]     # 8 - P5/32
  - [-1, 3, C3k2, [1024, True]]     # 9
  - [-1, 1, SPPF, [1024, 5]]        # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 7], 1, Concat, [1]]                   # 12 - Concat with P4
  - [-1, 3, C3k2, [512, False]]                 # 13
  - [-1, 1, CBAM, [512]]                        # 14 - Neck Attention
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 15
  - [[-1, 4], 1, Concat, [1]]                   # 16 - Concat with P3
  - [-1, 3, C3k2, [256, False]]                 # 17 (P3 Output)
  - [-1, 1, Conv, [256, 3, 2]]                  # 18
  - [[-1, 14], 1, Concat, [1]]                  # 19 - Concat with Layer 14
  - [-1, 3, C3k2, [512, False]]                 # 20 (P4 Output)
  - [-1, 1, Conv, [512, 3, 2]]                  # 21
  - [[-1, 10], 1, Concat, [1]]                  # 22 - Concat with P5
  - [-1, 3, C3k2, [1024, True]]                 # 23 (P5 Output)
  - [[17, 20, 23], 1, Detect, [nc]]             # Final Output Layer
"""

# Alternative: If you want to use SPDConv, use this YAML instead
SOCA_YOLO11M_YAML_WITH_SPD = """
nc: 4  # Number of sonar object classes

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0 - P1/2 (H/2, W/2, 64)
  - [-1, 1, SPDConv, [128, 1]]      # 1 - P2/4 (H/4, W/4, 128) SPD takes 64ch -> 256ch internally -> 128ch
  - [-1, 3, C3k2, [128, False]]     # 2
  - [-1, 1, SPDConv, [256, 1]]      # 3 - P3/8 (H/8, W/8, 256) SPD takes 128ch -> 512ch internally -> 256ch  
  - [-1, 6, C3k2, [256, False]]     # 4
  - [-1, 1, CBAM, [256]]            # 5 - Backbone Attention
  - [-1, 1, Conv, [512, 3, 2]]      # 6 - P4/16
  - [-1, 6, C3k2, [512, False]]     # 7
  - [-1, 1, Conv, [1024, 3, 2]]     # 8 - P5/32
  - [-1, 3, C3k2, [1024, True]]     # 9
  - [-1, 1, SPPF, [1024, 5]]        # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 7], 1, Concat, [1]]                   # 12 - Concat with P4 (512 + 512 = 1024)
  - [-1, 3, C3k2, [512, False]]                 # 13
  - [-1, 1, CBAM, [512]]                        # 14 - Neck Attention
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 15
  - [[-1, 4], 1, Concat, [1]]                   # 16 - Concat with P3 (512 + 256 = 768)
  - [-1, 3, C3k2, [256, False]]                 # 17 (P3 Output)
  - [-1, 1, Conv, [256, 3, 2]]                  # 18
  - [[-1, 14], 1, Concat, [1]]                  # 19 - Concat with Layer 14 (256 + 512 = 768)
  - [-1, 3, C3k2, [512, False]]                 # 20 (P4 Output)
  - [-1, 1, Conv, [512, 3, 2]]                  # 21
  - [[-1, 10], 1, Concat, [1]]                  # 22 - Concat with P5 (512 + 1024 = 1536)
  - [-1, 3, C3k2, [1024, True]]                 # 23 (P5 Output)
  - [[17, 20, 23], 1, Detect, [nc]]             # Final Output Layer
"""

# --- 4. EXECUTION ---
def main():
    # Check CUDA availability
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("No CUDA devices found. Training will use CPU (very slow).")
    print("="*60 + "\n")
    
    # Determine device
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ Training on CPU - this will be VERY slow!")
        print("  Consider using Google Colab, Kaggle, or a cloud GPU service.")
    
    # Save YAML - using the version WITHOUT SPDConv for now (more stable)
    with open("soca_yolo11m.yaml", "w") as f:
        f.write(SOCA_YOLO11M_YAML)
    
    # If you want to try with SPDConv, uncomment this:
    # with open("soca_yolo11m.yaml", "w") as f:
    #     f.write(SOCA_YOLO11M_YAML_WITH_SPD)

    # Roboflow Download
    print("\nDownloading dataset from Roboflow...")
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    dataset = project.version(1).download("yolov11")
    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"✓ Dataset downloaded to: {dataset.location}\n")

    # Build Model
    print("Building model...")
    model = YOLO("soca_yolo11m.yaml")
    
    print("\n" + "="*60)
    print("MODEL BUILT SUCCESSFULLY!")
    print("="*60 + "\n")

    # Train
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: 640")
    print(f"Epochs: 100")
    print("="*60 + "\n")
    
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=batch_size,
        device=device,
        project="runs/train",
        name="yolo11m_soca_final",
        verbose=True,
        plots=True,
        # Additional useful parameters for training
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save every 10 epochs
        cache=False,  # Set to True if you have enough RAM
        workers=8 if device != 'cpu' else 2,  # Data loading workers
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()