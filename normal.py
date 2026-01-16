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
        imgsz=640,
        epochs=500,
        patience=50,
        batch=batch_size,            # Auto-batch to maximize VRAM utilization
        project="runs/train",
        name="yolo11m_sonar", 
        verbose=True,
        plots=True # Always good to visualize the training curves
    )

if __name__ == '__main__':
    main()