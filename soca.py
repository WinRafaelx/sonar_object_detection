import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from roboflow import Roboflow
from constant import batch_size
import inspect

# --- 1. RESEARCH MODULES ---
class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution: Prevents information loss for tiny targets by folding pixels instead of deleting them.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2)

    def forward(self, x):
        # Slice and stack sub-regions
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: Helps the model focus on important stuff and ignore background noise.
    """
    def __init__(self, c1, c2=None, reduction=16, k=1, s=1, p=None, g=1, d=1, act=True):
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
        
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        x = x * self.sa(res)
        return self.proj(x)

# --- 2. GLOBAL REGISTRATION & MONKEY-PATCHING ---
# Register names in the tasks module so parse_model can find them
setattr(tasks, 'SPDConv', SPDConv)
setattr(tasks, 'CBAM', CBAM)

def patch_parse_model():
    try:
        source = inspect.getsource(tasks.parse_model)
        # Inject into base_modules and repeat_modules sets
        # We target 'A2C2f,' as it is usually near the end of the lists in YOLO11/v8
        new_source = source.replace("A2C2f,", "A2C2f, CBAM, SPDConv,")
        
        # Create execution context
        exec_globals = tasks.__dict__.copy()
        exec_globals.update({
            'SPDConv': SPDConv,
            'CBAM': CBAM,
            'torch': torch,
            'nn': nn,
            'inspect': inspect,
            'contextlib': __import__('contextlib'),
            'ast': __import__('ast'),
        })
        
        # Execute modified source
        exec(new_source, exec_globals)
        tasks.parse_model = exec_globals['parse_model']
        print("✓ Successfully patched YOLO model parser for custom modules.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser automatically: {e}")
        print("Continuing with default parser (may cause channel mismatch errors).")

patch_parse_model()

# --- 3. ARCHITECTURE DEFINITION (SOCA-YOLO11M with SPDConv) ---
SOCA_YOLO11M_YAML = """
nc: 4  # Number of sonar object classes

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0 - P1/2 
  - [-1, 1, SPDConv, [128, 1]]      # 1 - P2/4 
  - [-1, 3, C3k2, [128, False]]     # 2
  - [-1, 1, SPDConv, [256, 1]]      # 3 - P3/8  
  - [-1, 6, C3k2, [256, False]]     # 4
  - [-1, 1, CBAM, [256]]            # 5 - Backbone Attention
  - [-1, 1, Conv, [512, 3, 2]]      # 6 - P4/16
  - [-1, 6, C3k2, [512, False]]     # 7
  - [-1, 1, Conv, [1024, 3, 2]]     # 8 - P5/32
  - [-1, 3, C3k2, [1024, True]]     # 9
  - [-1, 1, SPPF, [1024, 5]]        # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 11
  - [[-1, 7], 1, Concat, [1]]                   # 12 
  - [-1, 3, C3k2, [512, False]]                 # 13
  - [-1, 1, CBAM, [512]]                        # 14 - Neck Attention
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 15
  - [[-1, 4], 1, Concat, [1]]                   # 16 
  - [-1, 3, C3k2, [256, False]]                 # 17 (P3 Output)
  - [-1, 1, Conv, [256, 3, 2]]                  # 18
  - [[-1, 14], 1, Concat, [1]]                  # 19 
  - [-1, 3, C3k2, [512, False]]                 # 20 (P4 Output)
  - [-1, 1, Conv, [512, 3, 2]]                  # 21
  - [[-1, 10], 1, Concat, [1]]                  # 22 
  - [-1, 3, C3k2, [1024, True]]                 # 23 (P5 Output)
  - [[17, 20, 23], 1, Detect, [nc]]             # Final Output Layer
"""

# --- 4. EXECUTION ---
def main():
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = 0
        print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ Training on CPU - this will be VERY slow!")
    print("="*60 + "\n")
    
    # Save Custom YAML
    yaml_path = "soca_yolo11m.yaml"
    with open(yaml_path, "w") as f:
        f.write(SOCA_YOLO11M_YAML)

    # Roboflow Download
    print("\nDownloading dataset from Roboflow...")
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    dataset = project.version(1).download("yolov11")
    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"✓ Dataset downloaded to: {dataset.location}\n")

    # Build Model with Transfer Learning
    print("Building model...")
    # Chains the custom blueprint with the pre-trained brain
    model = YOLO(yaml_path).load("yolo11m.pt") 
    
    print("\n" + "="*60)
    print("MODEL BUILT SUCCESSFULLY!")
    print("="*60 + "\n")

    # Train
    print("Starting training...")
    
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=batch_size,
        device=device,
        project="runs/train",
        name="yolo11m_soca_final",
        
        # --- SONAR-SPECIFIC TWEAKS ---
        rect=True,          # Don't squish long images
        optimizer='AdamW',  # Better for noisy data
        hsv_h=0.0,          # No fake color changes
        hsv_s=0.0,          
        hsv_v=0.0,          
        fliplr=0.5,         # Flip left/right is still okay
        # -----------------------------
        
        verbose=True,
        plots=True,
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        workers=8 if device != 'cpu' else 2,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()
