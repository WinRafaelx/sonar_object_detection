import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def yolo_to_xyxy(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Convert YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)"""
    center_x, center_y, width, height = bbox
    x1 = (center_x - width / 2) * img_width
    y1 = (center_y - height / 2) * img_height
    x2 = (center_x + width / 2) * img_width
    y2 = (center_y + height / 2) * img_height
    return np.array([x1, y1, x2, y2])

def load_yolo_labels(label_path: Path, img_width: int, img_height: int) -> List[Tuple[int, np.ndarray]]:
    """Load YOLO format labels and convert to (class_id, [x1, y1, x2, y2])"""
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = np.array([float(x) for x in parts[1:5]])
                xyxy = yolo_to_xyxy(bbox, img_width, img_height)
                labels.append((class_id, xyxy))
    return labels

def find_false_predictions(
    model: YOLO,
    data_yaml: str,
    split: str = 'val',
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    max_samples: int = 10
) -> List[Dict]:
    """Find false positives and false negatives in predictions"""
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get image directory
    split_key = 'val' if split == 'val' else ('test' if split == 'test' else 'train')
    img_dir = Path(data_config[split_key])
    if not img_dir.is_absolute():
        img_dir = Path(data_yaml).parent / img_dir
    
    label_dir = img_dir.parent.parent / split_key / 'labels'
    
    # Get all images
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    false_predictions = []
    
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # Get predictions
        pred_results = model.predict(
            str(img_path),
            conf=conf_threshold,
            verbose=False
        )
        
        # Get ground truth labels
        label_path = label_dir / (img_path.stem + '.txt')
        gt_labels = load_yolo_labels(label_path, img_width, img_height)
        
        # Extract predictions
        pred_boxes = []
        if len(pred_results) > 0 and pred_results[0].boxes is not None:
            boxes = pred_results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                pred_boxes.append((cls, box, conf))
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        false_positives = []
        false_negatives = []
        
        # Try to match each prediction with ground truth
        for pred_idx, (pred_cls, pred_box, pred_conf) in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_cls, gt_box) in enumerate(gt_labels):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
            else:
                # False positive: predicted but no matching ground truth
                false_positives.append((pred_cls, pred_box, pred_conf))
        
        # Find false negatives: ground truth with no matching prediction
        for gt_idx, (gt_cls, gt_box) in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                false_negatives.append((gt_cls, gt_box))
        
        # If we have false positives or false negatives, save this example
        if false_positives or false_negatives:
            false_predictions.append({
                'image_path': img_path,
                'image': img,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'all_predictions': pred_boxes,
                'ground_truth': gt_labels
            })
            
            if len(false_predictions) >= max_samples:
                break
    
    return false_predictions

def visualize_false_predictions(
    false_predictions: List[Dict],
    output_dir: Path,
    class_names: List[str]
):
    """Visualize and save false predictions"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for idx, sample in enumerate(false_predictions):
        img = sample['image']
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.axis('off')
        
        # Draw ground truth boxes (green)
        for gt_cls, gt_box in sample['ground_truth']:
            x1, y1, x2, y2 = gt_box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            class_name = class_names[gt_cls] if gt_cls < len(class_names) else f'Class{gt_cls}'
            ax.text(x1, y1-5, f'GT: {class_name}', 
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Draw all predictions (blue)
        for pred_cls, pred_box, pred_conf in sample['all_predictions']:
            x1, y1, x2, y2 = pred_box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            class_name = class_names[pred_cls] if pred_cls < len(class_names) else f'Class{pred_cls}'
            ax.text(x1, y1+15, f'Pred: {class_name} {pred_conf:.2f}', 
                   color='blue', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
        
        # Draw false positives (red) - predicted but not in ground truth
        for fp_cls, fp_box, fp_conf in sample['false_positives']:
            x1, y1, x2, y2 = fp_box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            class_name = class_names[fp_cls] if fp_cls < len(class_names) else f'Class{fp_cls}'
            ax.text(x1, y2+5, f'FP: {class_name} {fp_conf:.2f}', 
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
        
        # Draw false negatives (orange) - in ground truth but not predicted
        for fn_cls, fn_box in sample['false_negatives']:
            x1, y1, x2, y2 = fn_box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='orange', facecolor='none', linestyle=':'
            )
            ax.add_patch(rect)
            class_name = class_names[fn_cls] if fn_cls < len(class_names) else f'Class{fn_cls}'
            ax.text(x1, y2+5, f'FN: {class_name}', 
                   color='orange', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Ground Truth'),
            Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label='All Predictions'),
            Line2D([0], [0], color='red', lw=3, label='False Positive'),
            Line2D([0], [0], color='orange', lw=3, linestyle=':', label='False Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        title = f"Sample {idx+1}: {sample['image_path'].name}\n"
        title += f"FP: {len(sample['false_positives'])}, FN: {len(sample['false_negatives'])}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Save image
        output_path = output_dir / f"false_prediction_{idx+1:02d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved false prediction visualization: {output_path}")

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
        epochs=50,
        batch=16,
        project="runs/train",
        name="yolo11m_sonar", 
        verbose=True,
        plots=True # Always good to visualize the training curves
    )
    
    # 4. After training, find and visualize false predictions
    print("\n" + "="*60)
    print("Analyzing false predictions...")
    print("="*60)
    
    # Load best model
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_model_path.exists():
        # Fallback to last.pt if best.pt doesn't exist
        best_model_path = Path(results.save_dir) / "weights" / "last.pt"
    
    if best_model_path.exists():
        print(f"Loading best model from: {best_model_path}")
        trained_model = YOLO(str(best_model_path))
        
        # Load class names
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
        
        # Find false predictions
        false_predictions = find_false_predictions(
            model=trained_model,
            data_yaml=data_yaml,
            split='val',  # Use validation set
            iou_threshold=0.5,
            conf_threshold=0.25,
            max_samples=10
        )
        
        if false_predictions:
            # Create output directory
            output_dir = Path(results.save_dir) / "false_predictions"
            print(f"\nFound {len(false_predictions)} samples with false predictions")
            print(f"Saving visualizations to: {output_dir}")
            
            # Visualize and save
            visualize_false_predictions(false_predictions, output_dir, class_names)
            
            print(f"\n✓ Successfully saved {len(false_predictions)} false prediction visualizations")
        else:
            print("\nNo false predictions found (or validation set is empty)")
    else:
        print(f"\nWarning: Could not find trained model at {best_model_path}")

if __name__ == '__main__':
    main()