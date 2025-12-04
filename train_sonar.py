"""
Training Script: Train YOLO-Sonar model with specialized training protocol
"""
import os
import argparse
import yaml
from pathlib import Path
from pipeline.stage2_model_architecture.yolo_sonar import YOLOSonar
from pipeline.stage3_training.shape_iou_loss import ShapeIoULoss
from pipeline.stage3_training.nadir_masking import NadirMasker
from utils.logger import setup_logger

logger = setup_logger(__name__)


def download_dataset_from_roboflow(
    workspace: str = "object-detect-ury2h",
    project: str = "sonar_detect",
    version: int = 1,
    format_type: str = "yolov11"
) -> str:
    """
    Download dataset from Roboflow
    
    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number
        format_type: Export format (yolov11, yolo, etc.)
        
    Returns:
        Path to data.yaml file
        
    Note:
        Dataset is downloaded to the current working directory.
        The folder name is typically: {project}-{version}
        Example: sonar_detect-1/
    """
    try:
        from roboflow import Roboflow
        
        logger.info("Downloading dataset from Roboflow...")
        logger.info(f"Workspace: {workspace}, Project: {project}, Version: {version}")
        
        # Get API key from environment variable
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable not set. "
                "Please set it with: export ROBOFLOW_API_KEY=your_key"
            )
        
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        version_obj = project_obj.version(version)
        
        # Download dataset
        # Roboflow downloads to current directory with name: {project}-{version}
        dataset = version_obj.download(format_type)
        
        # Get path to data.yaml
        data_yaml = os.path.join(dataset.location, "data.yaml")
        
        logger.info(f"Dataset downloaded to: {dataset.location}")
        logger.info(f"Data YAML path: {data_yaml}")
        
        return data_yaml
        
    except ImportError:
        raise ImportError(
            "roboflow package not installed. Install with: pip install roboflow"
        )


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-Sonar model')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset YAML file (or use --download-roboflow)')
    parser.add_argument('--download-roboflow', action='store_true',
                       help='Download dataset from Roboflow instead of using --data')
    parser.add_argument('--roboflow-workspace', type=str, default='object-detect-ury2h',
                       help='Roboflow workspace name')
    parser.add_argument('--roboflow-project', type=str, default='sonar_detect',
                       help='Roboflow project name')
    parser.add_argument('--roboflow-version', type=int, default=1,
                       help='Roboflow dataset version')
    parser.add_argument('--model', type=str, default='yolo11m.pt', help='Base model weights')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='yolo_sonar', help='Run name')
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.download_roboflow:
        logger.info("Downloading dataset from Roboflow...")
        data_yaml = download_dataset_from_roboflow(
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=args.roboflow_version
        )
    elif args.data:
        data_yaml = args.data
        logger.info(f"Using dataset from: {data_yaml}")
    else:
        # Try to find existing dataset
        default_paths = [
            'sonar_detect-1/data.yaml',
            os.path.join(os.getcwd(), 'sonar_detect-1', 'data.yaml')
        ]
        for path in default_paths:
            if os.path.exists(path):
                data_yaml = path
                logger.info(f"Found existing dataset at: {data_yaml}")
                break
        else:
            raise ValueError(
                "No dataset specified. Use --data to specify path, "
                "or --download-roboflow to download from Roboflow"
            )
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize YOLO-Sonar model
    logger.info("Initializing YOLO-Sonar model...")
    model = YOLOSonar(config, model_path=args.model)
    
    # Training parameters
    train_params = {
        'data': data_yaml,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'project': args.project,
        'name': args.name,
        'verbose': True,
        'plots': True
    }
    
    # Note: Shape-IoU loss and Nadir masking integration requires
    # modifying Ultralytics training loop. For now, we train with standard loss.
    # Full integration would require custom training script that:
    # 1. Applies Stage 1 preprocessing to training images
    # 2. Uses ShapeIoULoss instead of standard IoU loss
    # 3. Applies NadirMasker to zero out loss in blind zones
    
    logger.info("Starting training...")
    logger.warning("Note: Shape-IoU and Nadir masking require custom training loop integration")
    
    # Train model
    results = model.train(**train_params)
    
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {results.save_dir}")


if __name__ == '__main__':
    main()

