"""
Main Pipeline Orchestrator

Coordinates all four stages of the Hybrid Acoustic-Vision Pipeline
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import cv2

from .stage1_signal_conditioning.processor import Stage1Processor
from .stage2_model_architecture.yolo_sonar import YOLOSonar
from .stage4_inference.sahi_slicing import SAHISlicer
from .stage4_inference.weighted_box_fusion import WBF
from .stage4_inference.geocoding import Geocoder
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PipelineOrchestrator:
    """
    Main orchestrator for the complete sonar detection pipeline
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize pipeline orchestrator
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            # Use default config
            default_config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
            with open(default_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize Stage 1
        self.stage1 = Stage1Processor(self.config)
        
        # Initialize Stage 2 (model)
        self.stage2 = None  # Will be initialized when model is loaded
        
        # Initialize Stage 4 components
        stage4_config = self.config.get('stage4', {})
        
        sahi_config = stage4_config.get('sahi', {})
        if sahi_config.get('enabled', True):
            self.sahi = SAHISlicer(
                slice_size=sahi_config.get('slice_size', 640),
                overlap_ratio=sahi_config.get('overlap_ratio', 0.25)
            )
        else:
            self.sahi = None
        
        wbf_config = stage4_config.get('wbf', {})
        if wbf_config.get('enabled', True):
            self.wbf = WBF(
                iou_threshold=wbf_config.get('iou_threshold', 0.5),
                skip_box_threshold=wbf_config.get('skip_box_threshold', 0.0001)
            )
        else:
            self.wbf = None
        
        geocoding_config = stage4_config.get('geocoding', {})
        if geocoding_config.get('enabled', True):
            self.geocoder = Geocoder(
                sonar_range=geocoding_config.get('sonar_range', 50.0),
                pixel_resolution=geocoding_config.get('pixel_resolution', 0.1)
            )
        else:
            self.geocoder = None
        
        logger.info("PipelineOrchestrator initialized")
    
    def load_model(self, model_path: str):
        """
        Load YOLO-Sonar model
        
        Args:
            model_path: Path to model weights
        """
        logger.info(f"Loading model from {model_path}")
        self.stage2 = YOLOSonar(self.config, model_path=model_path)
        logger.info("Model loaded successfully")
    
    def process_image(
        self,
        image_path: str,
        return_intermediates: bool = False
    ) -> Dict:
        """
        Process a single sonar image through the complete pipeline
        
        Args:
            image_path: Path to sonar image
            return_intermediates: If True, return intermediate results
            
        Returns:
            Dictionary containing:
            - processed_image: Stage 1 processed image
            - detections: Final detections
            - geocoded_detections: Geocoded detections (if enabled)
            - metadata: Stage 1 metadata (bottom_line, altitude, etc.)
            - intermediates: Intermediate results (if return_intermediates=True)
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        
        # Stage 1: Signal Conditioning
        logger.info("=== Stage 1: Signal Conditioning ===")
        processed_image, stage1_metadata = self.stage1.process(
            image,
            return_intermediates=return_intermediates
        )
        
        # Stage 2: Model Inference
        if self.stage2 is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("=== Stage 2: Model Inference ===")
        
        # Stage 4: SAHI Slicing (if enabled)
        if self.sahi:
            logger.info("=== Stage 4a: SAHI Slicing ===")
            slices = self.sahi.slice_image(processed_image)
            
            all_detections = []
            all_offsets = []
            
            for slice_img, offset in slices:
                # Run inference on slice
                results = self.stage2.predict(slice_img, verbose=False)
                
                # Extract detections
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    dets = []
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        dets.append([x1, y1, x2, y2, conf, cls])
                    
                    if dets:
                        all_detections.append(np.array(dets))
                        all_offsets.append(offset)
            
            # Merge detections from slices
            merged_detections = self.sahi.merge_detections(all_detections, all_offsets)
        else:
            # Direct inference without slicing
            results = self.stage2.predict(processed_image, verbose=False)
            
            merged_detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    merged_detections.append([x1, y1, x2, y2, conf, cls])
            
            merged_detections = np.array(merged_detections) if merged_detections else np.array([]).reshape(0, 6)
        
        # Stage 4: Weighted Box Fusion
        if self.wbf and len(merged_detections) > 0:
            logger.info("=== Stage 4b: Weighted Box Fusion ===")
            merged_detections = self.wbf.merge(merged_detections)
        
        # Stage 4: Geocoding
        geocoded_detections = None
        if self.geocoder and len(merged_detections) > 0:
            logger.info("=== Stage 4c: Geocoding ===")
            altitude = stage1_metadata.get('altitude')
            if altitude is not None:
                self.geocoder.image_height = processed_image.shape[0]
                geocoded_detections = self.geocoder.geocode_detections(
                    merged_detections,
                    altitude
                )
        
        # Prepare results
        results_dict = {
            'processed_image': processed_image,
            'detections': merged_detections,
            'geocoded_detections': geocoded_detections,
            'metadata': stage1_metadata
        }
        
        if return_intermediates:
            results_dict['intermediates'] = {
                'original_image': original_image,
                'stage1_processed': processed_image
            }
        
        logger.info(f"Pipeline complete. Detections: {len(merged_detections)}")
        
        return results_dict

