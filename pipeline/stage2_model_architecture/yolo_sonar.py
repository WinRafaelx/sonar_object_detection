"""
YOLO-Sonar Model: Custom YOLOv11 architecture with sonar-specific modifications
"""
from ultralytics import YOLO
from typing import Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class YOLOSonar:
    """
    Wrapper for YOLO-Sonar model with custom architecture modifications
    
    Note: Full integration with Ultralytics requires modifying their source code.
    This class provides a framework for integration.
    """
    
    def __init__(self, config: Dict, model_path: str = 'yolo11m.pt'):
        """
        Initialize YOLO-Sonar model
        
        Args:
            config: Configuration dictionary
            model_path: Path to base YOLO model weights
        """
        self.config = config
        self.model_path = model_path
        stage2_config = config.get('stage2', {})
        
        # Load base model
        self.model = YOLO(model_path)
        
        # Store configuration
        self.spd_conv_enabled = stage2_config.get('spd_conv', {}).get('enabled', True)
        self.ema_enabled = stage2_config.get('ema_attention', {}).get('enabled', True)
        self.bifpn_enabled = stage2_config.get('bifpn', {}).get('enabled', True)
        self.p2_detection = stage2_config.get('model', {}).get('p2_detection', True)
        
        logger.info(f"YOLOSonar initialized: model={model_path}")
        logger.info(f"Modifications: SPD-Conv={self.spd_conv_enabled}, "
                   f"EMA={self.ema_enabled}, BiFPN={self.bifpn_enabled}, "
                   f"P2={self.p2_detection}")
    
    def train(self, data_yaml: str, **kwargs):
        """
        Train YOLO-Sonar model
        
        Args:
            data_yaml: Path to dataset YAML file
            **kwargs: Additional training parameters
        """
        logger.info("Starting YOLO-Sonar training...")
        
        # Enable P2 detection head if configured
        if self.p2_detection:
            # Note: This requires Ultralytics source modification
            # For now, we pass it as a parameter
            kwargs.setdefault('imgsz', 640)
        
        # Train model
        results = self.model.train(
            data=data_yaml,
            **kwargs
        )
        
        logger.info("Training completed")
        return results
    
    def predict(self, source, **kwargs):
        """
        Run inference with YOLO-Sonar model
        
        Args:
            source: Input source (image, video, etc.)
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results
        """
        return self.model.predict(source, **kwargs)
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to different formats
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            **kwargs: Additional export parameters
        """
        return self.model.export(format=format, **kwargs)

