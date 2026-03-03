import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
from constant import batch_size

def merge_datasets():
    combined_path = Path("combined_dataset")
    combined_path.mkdir(exist_ok=True)
    
    # Define classes (merged from both datasets)
    # Roboflow: ['aircraft', 'fish', 'other', 'shipwreck'] (0-3)
    # Kaggle: ['cylinder', 'manta', 'rectangle', 'triangle'] (0-3)
    # Combined: ['aircraft', 'fish', 'other', 'shipwreck', 'cylinder', 'manta', 'rectangle', 'triangle'] (0-7)
    
    class_map_roboflow = {i: i for i in range(4)}
    class_map_kaggle = {i: i + 4 for i in range(4)}
    
    names = [
        'aircraft', 'fish', 'other', 'shipwreck',  # Roboflow 0-3
        'cylinder', 'manta', 'rectangle', 'triangle' # Kaggle 4-7
    ]
    
    for split in ["train", "val", "test"]:
        (combined_path / split / "images").mkdir(parents=True, exist_ok=True)
        (combined_path / split / "labels").mkdir(parents=True, exist_ok=True)

    def process_split(src_images, src_labels, dest_split, class_map):
        if not os.path.exists(src_images):
            print(f"Warning: {src_images} not found.")
            return
            
        for img_file in os.listdir(src_images):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            # Copy image
            shutil.copy2(os.path.join(src_images, img_file), combined_path / dest_split / "images" / img_file)
            
            # Process label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = os.path.join(src_labels, label_file)
            dest_label_path = combined_path / dest_split / "labels" / label_file
            
            if os.path.exists(src_label_path):
                with open(src_label_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        new_class_id = class_map.get(class_id)
                        if new_class_id is not None:
                            parts[0] = str(new_class_id)
                            new_lines.append(" ".join(parts) + "\n")
                
                with open(dest_label_path, 'w') as f:
                    f.writelines(new_lines)

    # 1. Process Roboflow
    print("Processing Roboflow dataset...")
    rf_base = Path("sonar_detect-1")
    process_split(rf_base / "train/images", rf_base / "train/labels", "train", class_map_roboflow)
    process_split(rf_base / "valid/images", rf_base / "valid/labels", "val", class_map_roboflow)
    process_split(rf_base / "test/images", rf_base / "test/labels", "test", class_map_roboflow)

    # 2. Process Kaggle
    print("Processing Kaggle dataset...")
    kg_base = Path("data/shapes_dataset_con/dataset")
    process_split(kg_base / "train/images", kg_base / "train/labels", "train", class_map_kaggle)
    process_split(kg_base / "val/images", kg_base / "val/labels", "val", class_map_kaggle)
    process_split(kg_base / "test/images", kg_base / "test/labels", "test", class_map_kaggle)

    # Create data.yaml
    data_yaml = {
        'path': str(combined_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(names),
        'names': names
    }
    
    with open("combined_data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)
    
    return "combined_data.yaml"

def main():
    yaml_path = merge_datasets()
    
    print(f"Dataset prepared at {yaml_path}")
    
    # Initialize YOLOv11m
    model = YOLO("yolo11m.pt")
    
    # Start training (dry run for 1 epoch to verify)
    print("Starting verification training (1 epoch)...")
    model.train(
        data=yaml_path,
        imgsz=640,
        epochs=1,  # Verification only
        batch=batch_size,
        project="runs/train_combined",
        name="yolo11m_combined_verify",
        verbose=True
    )
    
    print("\nVerification complete! You can now run this script with more epochs on your training machine.")

if __name__ == "__main__":
    main()
