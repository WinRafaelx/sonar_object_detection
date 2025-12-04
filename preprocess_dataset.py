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
    config_path: str = 'config/pipeline_config.yaml',
    test_mode: bool = False
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
        image_files_jpg = list(original_img_dir.glob('*.jpg'))
        image_files_png = list(original_img_dir.glob('*.png'))
        image_files = image_files_jpg + image_files_png
        
        logger.info(f"Found {len(image_files_jpg)} .jpg files and {len(image_files_png)} .png files")
        logger.info(f"Total: {len(image_files)} images to process")
        
        # Test mode: process only first image
        if test_mode and len(image_files) > 0:
            logger.info("TEST MODE: Processing only first image")
            image_files = image_files[:1]
        
        if len(image_files) == 0:
            logger.error(f"No image files found in {original_img_dir}")
            logger.error(f"Directory contents: {list(original_img_dir.iterdir())[:10]}")  # Show first 10 items
            if split in ['train', 'val']:
                raise FileNotFoundError(
                    f"No image files found for required split '{split}' in directory: {original_img_dir}"
                )
            continue
        
        processed_count = 0
        failed_count = 0
        first_error = None
        error_samples = []  # Store first 3 errors for reporting
        
        # Process first image with detailed logging to catch issues early
        if len(image_files) > 0:
            logger.info(f"Processing first image as test: {image_files[0].name}")
        
        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            try:
                # Load image
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    error_msg = f"Could not load image (cv2 returned None): {img_file}"
                    if failed_count < 3:
                        error_samples.append(error_msg)
                    logger.warning(error_msg)
                    failed_count += 1
                    if first_error is None:
                        first_error = error_msg
                    continue
                
                if image.size == 0:
                    error_msg = f"Loaded empty image: {img_file}"
                    if failed_count < 3:
                        error_samples.append(error_msg)
                    logger.warning(error_msg)
                    failed_count += 1
                    if first_error is None:
                        first_error = error_msg
                    continue
                
                # Ensure image is 2D (remove channel dimension if present)
                if len(image.shape) == 3:
                    # If it's grayscale with channel dimension, squeeze it
                    if image.shape[2] == 1:
                        image = image.squeeze(axis=2)
                    else:
                        # Convert to grayscale if it's color
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Process with Stage 1
                try:
                    # Call process and capture the result
                    result = processor.process(image, return_intermediates=False)
                    
                    # Debug: log what we got
                    logger.debug(f"processor.process() returned: type={type(result)}, "
                               f"is_tuple={isinstance(result, tuple)}, "
                               f"len={len(result) if hasattr(result, '__len__') else 'N/A'}")
                    
                    # Ensure we unpack exactly 2 values
                    if isinstance(result, tuple):
                        if len(result) == 2:
                            processed_image, metadata = result
                        else:
                            raise ValueError(
                                f"processor.process() returned tuple with {len(result)} values, expected 2. "
                                f"Result: {result[:3] if len(result) > 3 else result}"
                            )
                    else:
                        raise ValueError(
                            f"processor.process() returned non-tuple: type={type(result)}, value={result}"
                        )
                except ValueError as e:
                    # Re-raise ValueError with more context
                    error_msg = f"Stage 1 processing failed for {img_file.name}: {str(e)}"
                    if failed_count < 3:
                        error_samples.append(error_msg)
                        logger.error(error_msg, exc_info=True)
                    else:
                        logger.error(error_msg)
                    failed_count += 1
                    if first_error is None:
                        first_error = error_msg
                    continue
                except Exception as e:
                    error_msg = f"Stage 1 processing failed for {img_file.name}: {type(e).__name__}: {str(e)}"
                    if failed_count < 3:
                        error_samples.append(error_msg)
                        logger.error(error_msg, exc_info=True)
                    else:
                        logger.error(error_msg)
                    failed_count += 1
                    if first_error is None:
                        first_error = error_msg
                    continue
                
                # Convert to uint8 for saving
                try:
                    processed_uint8 = (processed_image * 255).astype(np.uint8)
                except Exception as e:
                    error_msg = f"Image conversion failed for {img_file.name}: {str(e)}"
                    if failed_count < 3:
                        error_samples.append(error_msg)
                        logger.error(error_msg, exc_info=True)
                    failed_count += 1
                    if first_error is None:
                        first_error = error_msg
                    continue
                
                # Save preprocessed image
                output_img_path = output_img_dir / img_file.name
                try:
                    success = cv2.imwrite(str(output_img_path), processed_uint8)
                    if not success:
                        error_msg = f"Failed to save processed image: {output_img_path}"
                        if failed_count < 3:
                            error_samples.append(error_msg)
                        logger.warning(error_msg)
                        failed_count += 1
                        if first_error is None:
                            first_error = error_msg
                        continue
                except Exception as e:
                    error_msg = f"Exception saving image {img_file.name}: {str(e)}"
                    if failed_count < 3:
                        error_samples.append(error_msg)
                        logger.error(error_msg, exc_info=True)
                    failed_count += 1
                    if first_error is None:
                        first_error = error_msg
                    continue
                
                processed_count += 1
                
            except Exception as e:
                error_msg = f"Unexpected error processing {img_file.name}: {str(e)}"
                if failed_count < 3:
                    error_samples.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                else:
                    logger.error(error_msg)
                failed_count += 1
                if first_error is None:
                    first_error = error_msg
        
        logger.info(f"{split}: Processed {processed_count}, Failed {failed_count}")
        
        if processed_count == 0:
            error_msg = (
                f"No images processed for {split}! "
                f"Found {len(image_files)} image files, but all failed to process.\n"
            )
            if first_error:
                error_msg += f"First error encountered: {first_error}\n"
            if error_samples:
                error_msg += f"Sample errors (first {len(error_samples)}):\n"
                for i, err in enumerate(error_samples, 1):
                    error_msg += f"  {i}. {err}\n"
            error_msg += "Check logs above for full error details."
            
            logger.error(error_msg)
            print(f"\n{'='*60}")
            print(f"ERROR: {error_msg}")
            print(f"{'='*60}\n")
            
            if split in ['train', 'val']:
                raise ValueError(error_msg)
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
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only first image from each split')
    
    args = parser.parse_args()
    
    # Preprocess dataset
    new_data_yaml = preprocess_dataset(
        data_yaml_path=args.data,
        output_dir=args.output,
        config_path=args.config,
        test_mode=args.test
    )
    
    logger.info(f"\nNext step: Train model with preprocessed dataset:")
    logger.info(f"python train_sonar.py --data {new_data_yaml}")


if __name__ == '__main__':
    main()

