"""
Simple Usage Example: How to use the pipeline programmatically
"""
import numpy as np
import cv2
from pipeline.pipeline_orchestrator import PipelineOrchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)


def example_basic_usage():
    """Example: Basic pipeline usage"""
    print("=== Example: Basic Pipeline Usage ===\n")
    
    # 1. Initialize pipeline with default config
    orchestrator = PipelineOrchestrator()
    
    # 2. Load trained model
    orchestrator.load_model('runs/train/yolo_sonar/weights/best.pt')
    
    # 3. Process an image
    results = orchestrator.process_image('path/to/sonar_image.png')
    
    # 4. Access results
    print(f"Found {len(results['detections'])} detections")
    print(f"Processed image shape: {results['processed_image'].shape}")
    print(f"Bottom line detected: {results['metadata']['bottom_line'] is not None}")
    
    if results['geocoded_detections'] is not None:
        print(f"Geocoded {len(results['geocoded_detections'])} detections")


def example_stage1_only():
    """Example: Use Stage 1 preprocessing only"""
    print("\n=== Example: Stage 1 Only ===\n")
    
    from pipeline.stage1_signal_conditioning.processor import Stage1Processor
    import yaml
    
    # Load config
    with open('config/pipeline_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = Stage1Processor(config)
    
    # Load image
    image = cv2.imread('path/to/sonar_image.png', cv2.IMREAD_GRAYSCALE)
    
    # Process
    processed, metadata = processor.process(image)
    
    print(f"Original: {image.shape}, Processed: {processed.shape}")
    print(f"Altitude: {np.mean(metadata['altitude']):.2f} pixels")


def example_custom_config():
    """Example: Use custom configuration"""
    print("\n=== Example: Custom Configuration ===\n")
    
    custom_config = {
        'stage1': {
            'viterbi': {
                'enabled': True,
                'max_jump': 15,  # Custom value
                'intensity_threshold': 0.4
            },
            'wavelet': {
                'enabled': True,
                'wavelet': 'haar',  # Different wavelet
                'threshold_multiplier': 0.6
            },
            'normalization': {
                'enabled': True,
                'log_transform': True,
                'clahe': {
                    'enabled': True,
                    'clip_limit': 3.0,
                    'tile_grid_size': [16, 16]
                }
            }
        },
        'stage4': {
            'sahi': {
                'enabled': True,
                'slice_size': 640,
                'overlap_ratio': 0.3  # More overlap
            },
            'wbf': {
                'enabled': True,
                'iou_threshold': 0.6
            },
            'geocoding': {
                'enabled': True,
                'sonar_range': 100.0,  # 100m range
                'pixel_resolution': 0.05  # 5cm per pixel
            }
        }
    }
    
    orchestrator = PipelineOrchestrator(config_dict=custom_config)
    print("Pipeline initialized with custom configuration")


def example_individual_components():
    """Example: Use individual components"""
    print("\n=== Example: Individual Components ===\n")
    
    from pipeline.stage1_signal_conditioning.viterbi_bottom_tracking import ViterbiBottomTracker
    from pipeline.stage1_signal_conditioning.wavelet_denoising import WaveletDenoiser
    from pipeline.stage4_inference.sahi_slicing import SAHISlicer
    
    # Use Viterbi tracker
    tracker = ViterbiBottomTracker(max_jump=10, intensity_threshold=0.3)
    image = np.random.rand(640, 1280)  # Example image
    bottom_line, blind_zone_mask = tracker.detect_bottom_line(image)
    print(f"Detected bottom line: {len(bottom_line)} points")
    
    # Use wavelet denoiser
    denoiser = WaveletDenoiser(wavelet='db4', mode='soft')
    denoised = denoiser.denoise(image)
    print(f"Denoised image: {denoised.shape}")
    
    # Use SAHI slicer
    slicer = SAHISlicer(slice_size=640, overlap_ratio=0.25)
    slices = slicer.slice_image(image)
    print(f"Created {len(slices)} slices")


if __name__ == '__main__':
    print("Pipeline Usage Examples\n")
    print("Note: These are code examples. Uncomment and modify paths to run.\n")
    
    # Uncomment to run examples:
    # example_basic_usage()
    # example_stage1_only()
    # example_custom_config()
    # example_individual_components()
    
    print("\nSee README_PIPELINE.md for detailed documentation.")

