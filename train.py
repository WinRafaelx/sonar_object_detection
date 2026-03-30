import os
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from roboflow import Roboflow
import inspect

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the parent directory to sys.path to import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constant import batch_size

# --- 1. RESEARCH & REGISTER MODULES ---
# Import custom modules for registration
try:
    from .custom_modules import SonarSPDConv, CoordAtt
except (ImportError, ValueError):
    from custom_modules import SonarSPDConv, CoordAtt

# Global Registration so parse_model can find them
setattr(tasks, 'SonarSPDConv', SonarSPDConv)
setattr(tasks, 'CoordAtt', CoordAtt)

def patch_parse_model():
    """
    Monkey-patches the YOLO parse_model function to correctly handle custom modules.
    Ensures that custom modules are added to the list of 'known' modules that 
    the YOLO engine tracks for channel counting and scaling.
    """
    try:
        source = inspect.getsource(tasks.parse_model)
        # In YOLOv11, we inject our modules into the 'Conv-like' list for channel scaling
        if "SonarSPDConv" not in source:
            new_source = source.replace("A2C2f,", "A2C2f, SonarSPDConv, CoordAtt,", 1)
            
            # Create execution context
            exec_globals = tasks.__dict__.copy()
            exec_globals.update({
                'SonarSPDConv': SonarSPDConv,
                'CoordAtt': CoordAtt,
                'torch': torch,
                'nn': nn,
                'inspect': inspect,
            })
            
            # Execute modified source and replace the function
            exec(new_source, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
            print("✓ Successfully patched YOLO model parser for custom modules.")
        else:
            print("✓ YOLO model parser already patched.")
    except Exception as e:
        print(f"⚠ Warning: Could not patch model parser automatically: {e}")

patch_parse_model()

def train_sonar():
    # 1. Roboflow Setup
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    version = project.version(1)
    
    # Returns a dataset object containing the absolute path to the download
    dataset = version.download("yolov11")
    data_yaml = os.path.join(dataset.location, "data.yaml")

    # 2. Initialize the model from your custom YAML structure
    model_yaml = os.path.join(os.path.dirname(__file__), "yolo11-sonar.yaml")
    model = YOLO(model_yaml)

    # 3. Load weights from standard YOLO11
    model.load("yolo11n.pt") 

    # 4. Start Training
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu', 
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        # Sonar-specific tweaks
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
    )

if __name__ == "__main__":
    train_sonar()
