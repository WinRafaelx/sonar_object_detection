"""
Inference Script: Run complete pipeline on sonar images
"""
import os
import argparse
from pathlib import Path
import yaml
from pipeline.pipeline_orchestrator import PipelineOrchestrator
from utils.visualization import visualize_detections, visualize_stage1_output
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run sonar object detection inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input sonar image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    orchestrator = PipelineOrchestrator(config_path=args.config)
    orchestrator.load_model(args.model)
    
    # Process image
    logger.info(f"Processing image: {args.image}")
    results = orchestrator.process_image(args.image, return_intermediates=True)
    
    # Save results
    import numpy as np
    
    # Save detections
    detections_file = output_dir / 'detections.txt'
    if len(results['detections']) > 0:
        np.savetxt(detections_file, results['detections'], 
                  fmt='%.2f', header='x1 y1 x2 y2 conf cls')
        logger.info(f"Saved {len(results['detections'])} detections to {detections_file}")
    else:
        logger.info("No detections found")
    
    # Save geocoded detections if available
    if results['geocoded_detections'] is not None:
        geocoded_file = output_dir / 'geocoded_detections.txt'
        np.savetxt(geocoded_file, results['geocoded_detections'],
                  fmt='%.4f', header='x1 y1 x2 y2 conf cls ground_x ground_y')
        logger.info(f"Saved geocoded detections to {geocoded_file}")
    
    # Visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        
        # Stage 1 visualization
        stage1_viz = output_dir / 'stage1_processing.png'
        visualize_stage1_output(
            results['intermediates']['original_image'],
            results['processed_image'],
            results['metadata'].get('bottom_line'),
            save_path=stage1_viz
        )
        logger.info(f"Saved Stage 1 visualization to {stage1_viz}")
        
        # Detection visualization
        if len(results['detections']) > 0:
            det_viz = output_dir / 'detections.png'
            visualize_detections(
                results['processed_image'],
                results['detections'],
                save_path=det_viz
            )
            logger.info(f"Saved detection visualization to {det_viz}")
    
    logger.info("Inference complete!")


if __name__ == '__main__':
    main()

