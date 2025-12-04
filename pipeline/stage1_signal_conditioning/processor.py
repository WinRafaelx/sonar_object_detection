"""
Stage 1 Processor: Integrates all signal conditioning components
"""
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

from .viterbi_bottom_tracking import ViterbiBottomTracker
from .wavelet_denoising import WaveletDenoiser
from .normalization import Normalizer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Stage1Processor:
    """
    Main processor for Stage 1: Physics-Based Signal Conditioning
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Stage 1 processor from configuration
        
        Args:
            config: Configuration dictionary for Stage 1
        """
        self.config = config
        stage1_config = config.get('stage1', {})
        
        # Initialize components based on config
        viterbi_config = stage1_config.get('viterbi', {})
        if viterbi_config.get('enabled', True):
            self.viterbi = ViterbiBottomTracker(
                max_jump=viterbi_config.get('max_jump', 10),
                intensity_threshold=viterbi_config.get('intensity_threshold', 0.3)
            )
        else:
            self.viterbi = None
        
        wavelet_config = stage1_config.get('wavelet', {})
        if wavelet_config.get('enabled', True):
            self.wavelet = WaveletDenoiser(
                wavelet=wavelet_config.get('wavelet', 'db4'),
                mode=wavelet_config.get('mode', 'soft'),
                threshold_multiplier=wavelet_config.get('threshold_multiplier', 0.5)
            )
        else:
            self.wavelet = None
        
        norm_config = stage1_config.get('normalization', {})
        clahe_config = norm_config.get('clahe', {})
        if norm_config.get('enabled', True):
            self.normalizer = Normalizer(
                log_transform=norm_config.get('log_transform', True),
                clahe_enabled=clahe_config.get('enabled', True),
                clahe_clip_limit=clahe_config.get('clip_limit', 2.0),
                clahe_tile_grid_size=tuple(clahe_config.get('tile_grid_size', [8, 8]))
            )
        else:
            self.normalizer = None
        
        logger.info("Stage1Processor initialized")
    
    def process(
        self,
        sonar_image: np.ndarray,
        return_intermediates: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process sonar image through Stage 1 pipeline
        
        Args:
            sonar_image: Raw 2D sonar image
            return_intermediates: If True, return intermediate results
            
        Returns:
            processed_image: Final processed image
            metadata: Dictionary containing bottom_line, blind_zone_mask, altitude, etc.
        """
        logger.info(f"Processing Stage 1: Input shape {sonar_image.shape}")
        
        # Store original for reference
        original = sonar_image.copy()
        
        # Step 1: Normalize to [0, 1] for processing
        if sonar_image.max() > 1.0:
            sonar_image = sonar_image.astype(np.float32) / sonar_image.max()
        else:
            sonar_image = sonar_image.astype(np.float32)
        
        # Step 2: Viterbi bottom tracking
        bottom_line = None
        blind_zone_mask = None
        altitude = None
        
        if self.viterbi:
            logger.debug("Running Viterbi bottom tracking...")
            bottom_line, blind_zone_mask = self.viterbi.detect_bottom_line(sonar_image)
            altitude = self.viterbi.get_altitude(bottom_line, sonar_image.shape[0])
        else:
            logger.debug("Viterbi bottom tracking disabled")
        
        # Step 3: Wavelet denoising
        if self.wavelet:
            logger.debug("Applying wavelet denoising...")
            sonar_image = self.wavelet.denoise(sonar_image)
        else:
            logger.debug("Wavelet denoising disabled")
        
        # Step 4: Dynamic range normalization
        if self.normalizer:
            logger.debug("Applying normalization...")
            sonar_image = self.normalizer.normalize(sonar_image)
        else:
            logger.debug("Normalization disabled")
        
        # Prepare metadata
        metadata = {
            'bottom_line': bottom_line,
            'blind_zone_mask': blind_zone_mask,
            'altitude': altitude,
            'original_shape': original.shape
        }
        
        if return_intermediates:
            metadata['intermediate_images'] = {
                'after_wavelet': sonar_image.copy() if self.wavelet else None,
                'after_normalization': sonar_image.copy()
            }
        
        logger.info("Stage 1 processing complete")
        
        return sonar_image, metadata

