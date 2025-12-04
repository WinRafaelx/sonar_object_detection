"""
Preprocess Dataset: Apply Stage 1 preprocessing to all training/validation/test images
"""
import os
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pipeline.stage1_signal_conditioning.processor import Stage1Processor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def preprocess_dataset(
    data_yaml_path: str,
    output_dir: str,
    config_path: str = 'config/pipeline_config.yaml'
):
    """
    Preprocess entire dataset using Stage 1 pipeline
    
    Args:
        data_yaml_path: Path to original data.yaml
        output_dir: Directory to save preprocessed images
        config_path: Path to pipeline configuration
    """
    # Load original data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Load pipeline config
    with open(config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)
    
    # Initialize Stage 1 processor
    logger.info("Initializing Stage 1 processor...")
    processor = Stage1Processor(pipeline_config)
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    new_data_config = data_config.copy()
    
    # Track which splits were successfully processed
    processed_splits = []
    
    for split in splits:
        if split not in data_config:
            logger.warning(f"Split '{split}' not found in data.yaml, skipping...")
            continue
        
        logger.info(f"\n=== Processing {split} split ===")
        
        # Get original image directory
        # data.yaml paths are relative to the yaml file location
        # Make sure data_yaml_dir is absolute
        data_yaml_dir = Path(data_yaml_path).resolve().parent
        split_path = data_config[split]
        
        # Handle relative paths
        # The path in data.yaml might already include 'images' or might not
        original_img_dir = None
        
        if os.path.isabs(split_path):
            # Absolute path
            original_img_dir = Path(split_path)
            if not original_img_dir.exists():
                # Try adding 'images' if it doesn't exist
                original_img_dir = Path(split_path) / 'images'
        else:
            # Relative path from data.yaml location
            # Try multiple path resolution strategies
            
            # Strategy 1: Use path as-is from data.yaml location
            # (e.g., ../train/images from sonar_detect-1/ -> sonar_object_detection/train/images)
            candidate = (data_yaml_dir / split_path).resolve()
            if candidate.exists() and candidate.is_dir():
                original_img_dir = candidate
            
            # Strategy 2: If path starts with ../, also try without the ../
            # (e.g., train/images from sonar_detect-1/ -> sonar_detect-1/train/images)
            if original_img_dir is None and split_path.startswith('../'):
                # Remove ../ prefix and try
                relative_path = split_path[3:]  # Remove '../'
                candidate = (data_yaml_dir / relative_path).resolve()
                if candidate.exists() and candidate.is_dir():
                    original_img_dir = candidate
            
            # Strategy 3: Try just the split name with 'images'
            # (e.g., train/images)
            if original_img_dir is None:
                # Extract split name (train, val, test) and try that
                split_name = split
                candidate = (data_yaml_dir / split_name / 'images').resolve()
                if candidate.exists() and candidate.is_dir():
                    original_img_dir = candidate
            
            # Strategy 4: Try without 'images' suffix (in case path doesn't include it)
            if original_img_dir is None:
                candidate = (data_yaml_dir / split_path).resolve()
                if candidate.exists() and candidate.is_dir():
                    original_img_dir = candidate
        
        # Final check - make sure it exists and is a directory
        if original_img_dir is None or not original_img_dir.exists() or not original_img_dir.is_dir():
            logger.error(f"Image directory not found for {split}")
            logger.error(f"  Split path in data.yaml: {split_path}")
            logger.error(f"  Data yaml location (absolute): {data_yaml_dir}")
            
            # Try to find the actual location by checking common patterns
            possible_paths = [
                (data_yaml_dir / split_path).resolve(),
                (data_yaml_dir / split / 'images').resolve(),
                (data_yaml_dir.parent / split_path).resolve(),
            ]
            if split_path.startswith('../'):
                relative_path = split_path[3:]
                possible_paths.append((data_yaml_dir / relative_path).resolve())
            
            logger.error(f"  Tried paths:")
            for pp in possible_paths:
                exists = pp.exists() if pp else False
                is_dir = pp.is_dir() if exists else False
                logger.error(f"    - {pp} (exists: {exists}, is_dir: {is_dir})")
            
            if split in ['train', 'val']:
                raise FileNotFoundError(
                    f"Required split '{split}' images not found. "
                    f"Split path in data.yaml: '{split_path}'. "
                    f"Please check that the path is correct relative to data.yaml location: {data_yaml_dir}"
                )
            continue
        
        logger.info(f"Found {split} images at: {original_img_dir}")
        
        # Create output directories
        output_img_dir = output_path / split / 'images'
        output_label_dir = output_path / split / 'labels'
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy labels (they don't need preprocessing)
        original_label_dir = original_img_dir.parent / 'labels'
        if original_label_dir.exists():
            import shutil
            for label_file in original_label_dir.glob('*.txt'):
                shutil.copy2(label_file, output_label_dir / label_file.name)
            logger.info(f"Copied {len(list(original_label_dir.glob('*.txt')))} label files")
        
        # Process images
        image_files = list(original_img_dir.glob('*.jpg')) + list(original_img_dir.glob('*.png'))
        logger.info(f"Found {len(image_files)} images to process")
        
        processed_count = 0
        failed_count = 0
        
        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            try:
                # Load image
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logger.warning(f"Could not load image: {img_file}")
                    failed_count += 1
                    continue
                
                # Process with Stage 1
                processed_image, metadata = processor.process(image)
                
                # Convert to uint8 for saving
                processed_uint8 = (processed_image * 255).astype(np.uint8)
                
                # Save preprocessed image
                output_img_path = output_img_dir / img_file.name
                cv2.imwrite(str(output_img_path), processed_uint8)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                failed_count += 1
        
        logger.info(f"{split}: Processed {processed_count}, Failed {failed_count}")
        
        if processed_count == 0:
            logger.warning(f"No images processed for {split}! Skipping in data.yaml")
            if split in ['train', 'val']:
                raise ValueError(f"Failed to process any images for required split: {split}")
            continue
        
        # Update data.yaml paths (relative to the new data.yaml location)
        # Since data.yaml will be at output_path/data.yaml, and images are at output_path/split/images
        # The relative path is: split/images
        new_data_config[split] = f"{split}/images"
        processed_splits.append(split)
    
    # Save new data.yaml
    new_data_yaml_path = output_path / 'data.yaml'
    
    # Ensure all required splits exist before saving
    required_splits = ['train', 'val']
    for split in required_splits:
        if split not in processed_splits:
            logger.error(f"Required split '{split}' was not processed! Cannot create data.yaml")
            raise ValueError(f"Missing required split: {split}")
    
    # Remove test split if it wasn't processed (optional)
    if 'test' in new_data_config and 'test' not in processed_splits:
        logger.warning("Test split not processed, removing from data.yaml")
        del new_data_config['test']
    
    with open(new_data_yaml_path, 'w') as f:
        yaml.dump(new_data_config, f, default_flow_style=False)
    
    # Verify the paths in data.yaml are correct
    logger.info("\n=== Verifying data.yaml paths ===")
    with open(new_data_yaml_path, 'r') as f:
        verify_config = yaml.safe_load(f)
    for split in ['train', 'val', 'test']:
        if split in verify_config:
            split_path = output_path / verify_config[split]
            if split_path.exists():
                logger.info(f"✓ {split}: {verify_config[split]} -> exists")
            else:
                logger.warning(f"✗ {split}: {verify_config[split]} -> NOT FOUND at {split_path}")
    
    logger.info(f"\n=== Preprocessing Complete ===")
    logger.info(f"Preprocessed dataset saved to: {output_path}")
    logger.info(f"New data.yaml: {new_data_yaml_path}")
    
    return str(new_data_yaml_path)


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset with Stage 1 pipeline')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to original data.yaml file')
    parser.add_argument('--output', type=str, default='dataset_preprocessed',
                       help='Output directory for preprocessed dataset')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    
    args = parser.parse_args()
    
    # Preprocess dataset
    new_data_yaml = preprocess_dataset(
        data_yaml_path=args.data,
        output_dir=args.output,
        config_path=args.config
    )
    
    logger.info(f"\nNext step: Train model with preprocessed dataset:")
    logger.info(f"python train_sonar.py --data {new_data_yaml}")


if __name__ == '__main__':
    main()

