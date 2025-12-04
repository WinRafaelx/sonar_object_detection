"""
Evaluation Script: Evaluate trained model on test set
"""
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    logger.info("=== Model Evaluation ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Split: {args.split}")
    
    # Load model
    logger.info("Loading model...")
    model = YOLO(args.model)
    
    # Run validation
    logger.info(f"Running evaluation on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True
    )
    
    # Print results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"mAP50: {results.box.map50:.4f}")
    logger.info(f"mAP50-95: {results.box.map:.4f}")
    logger.info(f"Precision: {results.box.mp:.4f}")
    logger.info(f"Recall: {results.box.mr:.4f}")
    
    # Class-wise results
    if hasattr(results.box, 'maps'):
        logger.info("\n=== Per-Class mAP50 ===")
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
        for i, (class_name, map50) in enumerate(zip(class_names, results.box.maps)):
            logger.info(f"  {class_name}: {map50:.4f}")
    
    logger.info(f"\nResults saved to: {results.save_dir}")


if __name__ == '__main__':
    main()

