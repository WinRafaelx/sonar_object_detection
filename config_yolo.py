import os
from ultralytics import YOLO
from roboflow import Roboflow
from constant import batch_size

def main():
    # 1. Roboflow Setup
    # Best Practice: Load this from env vars so you don't leak keys on GitHub
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("object-detect-ury2h").project("sonar_detect")
    version = project.version(1)
    
    # Returns a dataset object containing the absolute path to the download
    dataset = version.download("yolov11")

    # 2. Dynamic Pathing
    # The dataset object has a .location attribute. 
    # This is safer than hardcoding "Sonar Detect YOLOv11" because version names/folders can shift.
    data_yaml = os.path.join(dataset.location, "data.yaml")

    # 3. Train YOLOv11
    # 'yolo11m.pt' is the medium model. Good balance of speed/accuracy.
    model = YOLO("yolo11m.pt")

    results = model.train(
        data=data_yaml,
        imgsz=1024,          # High res for small objects
        epochs=100,
        batch=batch_size,            # Auto-batch to maximize A40 VRAM utilization
        overlap_mask=True,   # Helps if objects are close together
        augment=True,
        mosaic=1.0,          # Strong augmentation for small object variety
        mixup=0.1,           # Helps with noise robustness
        project="runs/train",
        name="yolo11m_sonar_high_res", 
        # Hyperparameters for small objects
        box=7.5,             # Increase box gain if localization is poor
        cls=0.5,             # Balance classification
    )

if __name__ == '__main__':
    main()