"""
Test Script: Test individual pipeline stages for debugging
"""
import argparse
import numpy as np
import cv2
from pathlib import Path
import yaml
from pipeline.stage1_signal_conditioning.processor import Stage1Processor
from utils.visualization import visualize_stage1_output
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_stage1(image_path: str, config_path: str = 'config/pipeline_config.yaml'):
    """Test Stage 1: Signal Conditioning"""
    logger.info("=== Testing Stage 1 ===")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    logger.info(f"Loaded image: {image.shape}")
    
    # Initialize processor
    processor = Stage1Processor(config)
    
    # Process
    processed, metadata = processor.process(image, return_intermediates=True)
    
    # Display results
    logger.info("Stage 1 Results:")
    logger.info(f"  Original shape: {image.shape}")
    logger.info(f"  Processed shape: {processed.shape}")
    logger.info(f"  Bottom line detected: {metadata['bottom_line'] is not None}")
    if metadata['bottom_line'] is not None:
        logger.info(f"  Mean altitude: {np.mean(metadata['altitude']):.2f} pixels")
    
    # Save visualization
    output_path = Path('test_output') / 'stage1_test.png'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    visualize_stage1_output(
        image,
        processed,
        metadata.get('bottom_line'),
        save_path=output_path
    )
    logger.info(f"Visualization saved to {output_path}")
    
    return processed, metadata


def main():
    parser = argparse.ArgumentParser(description='Test pipeline stages')
    parser.add_argument('--image', type=str, required=True, help='Test image path')
    parser.add_argument('--stage', type=str, default='1', choices=['1', '2', '3', '4', 'all'],
                       help='Stage to test')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    if args.stage in ['1', 'all']:
        test_stage1(args.image, args.config)
    
    if args.stage in ['2', 'all']:
        logger.warning("Stage 2 testing requires trained model. Use inference.py instead.")
    
    if args.stage in ['3', 'all']:
        logger.warning("Stage 3 testing requires training loop. Use train_sonar.py instead.")
    
    if args.stage in ['4', 'all']:
        logger.warning("Stage 4 testing requires model inference. Use inference.py instead.")
    
    logger.info("Testing complete!")


if __name__ == '__main__':
    main()

