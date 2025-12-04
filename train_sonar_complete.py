"""
Complete Training Script: Preprocess → Train → Evaluate
This script runs the complete pipeline: preprocessing, training, and evaluation
"""
import os
import argparse
import yaml
from pathlib import Path
from pipeline.stage2_model_architecture.yolo_sonar import YOLOSonar
from utils.logger import setup_logger

logger = setup_logger(__name__)


def download_dataset_from_roboflow(
    workspace: str = "object-detect-ury2h",
    project: str = "sonar_detect",
    version: int = 1,
    format_type: str = "yolov11"
) -> str:
    """Download dataset from Roboflow"""
    try:
        from roboflow import Roboflow
        
        logger.info("Downloading dataset from Roboflow...")
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set")
        
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        version_obj = project_obj.version(version)
        dataset = version_obj.download(format_type)
        
        data_yaml = os.path.join(dataset.location, "data.yaml")
        logger.info(f"Dataset downloaded to: {dataset.location}")
        return data_yaml
        
    except ImportError:
        raise ImportError("roboflow package not installed. Install with: pip install roboflow")


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: Preprocess → Train → Evaluate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete Pipeline Workflow:
  1. Preprocess dataset (Stage 1: Signal Conditioning)
  2. Train model (Stage 2: YOLO-Sonar)
  3. Evaluate model on test set

Example:
  python train_sonar_complete.py --data sonar_detect-1/data.yaml --preprocess
        """
    )
    
    # Dataset arguments
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset YAML file')
    parser.add_argument('--download-roboflow', action='store_true',
                       help='Download dataset from Roboflow')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess dataset with Stage 1 before training')
    parser.add_argument('--preprocessed-data', type=str, default=None,
                       help='Path to preprocessed dataset YAML (skip preprocessing)')
    
    # Training arguments
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                       help='Base model weights')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='yolo_sonar',
                       help='Run name')
    
    # Evaluation arguments
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Step 1: Get dataset
    if args.download_roboflow:
        logger.info("=== Step 1: Downloading Dataset ===")
        data_yaml = download_dataset_from_roboflow()
    elif args.preprocessed_data:
        logger.info("=== Step 1: Using Preprocessed Dataset ===")
        data_yaml = args.preprocessed_data
        logger.info(f"Using preprocessed dataset: {data_yaml}")
    elif args.data:
        data_yaml = args.data
        logger.info(f"Using dataset: {data_yaml}")
    else:
        # Try to find existing dataset
        default_paths = [
            'sonar_detect-1/data.yaml',
            os.path.join(os.getcwd(), 'sonar_detect-1', 'data.yaml')
        ]
        for path in default_paths:
            if os.path.exists(path):
                data_yaml = path
                logger.info(f"Found existing dataset: {data_yaml}")
                break
        else:
            raise ValueError(
                "No dataset specified. Use --data, --download-roboflow, "
                "or --preprocessed-data"
            )
    
    # Step 2: Preprocess dataset (if requested)
    if args.preprocess and not args.preprocessed_data:
        logger.info("\n=== Step 2: Preprocessing Dataset (Stage 1) ===")
        from preprocess_dataset import preprocess_dataset
        
        preprocessed_dir = Path(args.project) / args.name / 'preprocessed_dataset'
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        new_data_yaml = preprocess_dataset(
            data_yaml_path=data_yaml,
            output_dir=str(preprocessed_dir),
            config_path=args.config
        )
        data_yaml = new_data_yaml
        logger.info(f"Preprocessing complete. Using: {data_yaml}")
    
    # Step 3: Load configuration
    logger.info("\n=== Step 3: Loading Configuration ===")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 4: Initialize and train model
    logger.info("\n=== Step 4: Training Model (Stage 2) ===")
    model = YOLOSonar(config, model_path=args.model)
    
    train_params = {
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'project': args.project,
        'name': args.name,
        'verbose': True,
        'plots': True
    }
    
    logger.info("Starting training...")
    # Pass data_yaml as positional argument, other params as kwargs
    results = model.train(data_yaml, **train_params)
    
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    logger.info(f"Training complete! Best model: {best_model_path}")
    
    # Step 5: Evaluate (if requested)
    if args.eval:
        logger.info("\n=== Step 5: Evaluation ===")
        logger.info("Running evaluation on test set...")
        
        # Run validation
        val_results = model.model.val(data=data_yaml)
        
        logger.info("\n=== Evaluation Results ===")
        logger.info(f"mAP50: {val_results.box.map50:.4f}")
        logger.info(f"mAP50-95: {val_results.box.map:.4f}")
        logger.info(f"Precision: {val_results.box.mp:.4f}")
        logger.info(f"Recall: {val_results.box.mr:.4f}")
    
    logger.info("\n=== Pipeline Complete ===")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"\nTo run inference:")
    logger.info(f"python inference.py --image [image_path] --model {best_model_path} --visualize")


if __name__ == '__main__':
    main()

