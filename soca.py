import os
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from constant import batch_size
import inspect
import yaml
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# Helper to ensure Kaggle API can find credentials
if 'KAGGLE_API_TOKEN' in os.environ and 'KAGGLE_KEY' not in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']

# Check if Kaggle credentials exist (either as environment variables or as a file)
kaggle_creds_found = ('KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ) or os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))

import kaggle

# --- 1. RESEARCH MODULES ---
class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution: Prevents information loss for tiny targets by folding pixels instead of deleting them.
    Useful for sonar imagery where objects are often small and low-contrast.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # c1*4 because the forward pass stacks 4 sub-regions into the channel dimension
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2)

    def forward(self, x):
        # Slice and stack sub-regions (Space-to-Depth)
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: Channel and Spatial Attention.
    Helps the model focus on objects and ignore background sonar noise.
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
    """
    Monkey-patches the YOLO model parser to support CBAM and SPDConv as standard layers.
    This ensures that depth/width multipliers are correctly applied and channel counting is correct.
    """
    try:
        source = inspect.getsource(tasks.parse_model)
        
        # Verify A2C2f exists in source before replacing (for robustness)
        if 'A2C2f' not in source:
            print("⚠ Warning: 'A2C2f' not found in parse_model source. Using alternative patch point...")
            target = "C2fCIB," if "C2fCIB," in source else "C2f,"
            new_source = source.replace(target, f"{target} CBAM, SPDConv,")
        else:
            new_source = source.replace("A2C2f,", "A2C2f, CBAM, SPDConv,")
        
        # Create execution context using the tasks module's namespace
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
        
        # Execute modified source to re-define parse_model in our context
        exec(new_source, exec_globals)
        tasks.parse_model = exec_globals['parse_model']
        print("✓ Successfully patched YOLO model parser for custom modules.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser automatically: {e}")
        print("Continuing with default parser (may cause channel mismatch or repetition errors).")

patch_parse_model()

# --- 3. ARCHITECTURE DEFINITION (SOCA-YOLO11M with SPDConv) ---
SOCA_YOLO11M_YAML = """
# nc is injected dynamically in main()

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
    print("SOCA-YOLO TRAINING SYSTEM")
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
    
    # Kaggle Download
    dataset_path = './data/Combined_Dataset'
    if not os.path.exists(os.path.join(dataset_path, "data.yaml")):
        if not kaggle_creds_found:
            print("⚠ ERROR: Kaggle credentials not found! Place 'kaggle.json' in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY.")
            sys.exit(1)
        
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
    
    # Save Custom Model YAML with dynamic nc
    yaml_path = "soca_yolo11m.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"nc: {nc}\n" + SOCA_YOLO11M_YAML)
    
    print(f"✓ Dataset prepared at: {dataset_path} with nc={nc}\n")

    # Shared training parameters
    train_args = {
        'data': data_yaml,
        'imgsz': 640,
        'epochs': 5,
        'batch': batch_size,
        'device': device,
        'project': "runs/train",
        'rect': True,
        'optimizer': 'AdamW',
        'degrees': 10.0,
        'fliplr': 0.5,
        'flipud': 0.0,
        'mixup': 0.1,
        'scale': 0.5,
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.4,
        'verbose': True,
        'plots': True,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': 8 if device != 'cpu' else 2,
    }

    # --- RUN 1: BARE FROM SCRATCH ---
    print("\n" + "="*60)
    print("RUN 1: TRAINING FROM SCRATCH (NO PRETRAINED WEIGHTS)")
    print("="*60 + "\n")
    model_scratch = YOLO(yaml_path)
    results_scratch = model_scratch.train(name="yolo11m_soca_scratch", **train_args)

    # --- RUN 2: PRETRAINED MODEL ---
    print("\n" + "="*60)
    print("RUN 2: TRAINING WITH PRETRAINED WEIGHTS (TRANSFER LEARNING)")
    print("="*60 + "\n")
    # Load architecture from YAML, then inject weights from yolo11m.pt
    model_pretrained = YOLO(yaml_path).load("yolo11m.pt")
    results_pretrained = model_pretrained.train(name="yolo11m_soca_pretrained", **train_args)

    # --- FINAL COMPARISON ---
    print("\n" + "="*60)
    print("FINAL ACCURACY METRIC COMPARISON")
    print("="*60)
    
    def get_metrics(results):
        """Extracts metrics from a DetMetrics object."""
        names = results.names
        box = results.box
        ap50_per_class = box.ap50 # List of AP50 for each class
        
        # Ensure names are mapped correctly to class indices
        per_class = {names[i]: ap50_per_class[i] for i in range(len(names))}
        
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

    # 1. Overall Comparison Table
    print("\n[ OVERALL METRICS ]")
    header = f"{'Metric':<15} | {'Scratch':<15} | {'Pretrained':<15} | {'Diff':<15}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    overall_scratch = m_scratch["overall"]
    overall_pretrained = m_pretrained["overall"]
    for key in overall_scratch.keys():
        s_val = overall_scratch[key]
        p_val = overall_pretrained[key]
        diff = p_val - s_val
        print(f"{key:<15} | {s_val:<15.4f} | {p_val:<15.4f} | {diff:<+15.4f}")
    print(separator)

    # 2. Per-Class Comparison Table (mAP50)
    print("\n[ PER-CLASS mAP50 ]")
    class_header = f"{'Class Name':<15} | {'Scratch':<15} | {'Pretrained':<15} | {'Diff':<15}"
    class_separator = "-" * len(class_header)
    print(class_header)
    print(class_separator)

    pc_scratch = m_scratch["per_class"]
    pc_pretrained = m_pretrained["per_class"]
    for cls_name in pc_scratch.keys():
        s_val = pc_scratch[cls_name]
        p_val = pc_pretrained.get(cls_name, 0)
        diff = p_val - s_val
        print(f"{cls_name:<15} | {s_val:<15.4f} | {p_val:<15.4f} | {diff:<+15.4f}")
    print(class_separator)

    print("\nComparison completed. Check 'runs/train' for detailed plots and confusion matrices.")

if __name__ == '__main__':
    main()
