"""
Dynamic Range Normalization

Purpose: Standardize acoustic returns for CNN input
- Logarithmic transformation to compress high-intensity signals
- CLAHE for local contrast enhancement
"""
import numpy as np
from skimage import exposure
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Normalizer:
    """
    Implements dynamic range normalization for sonar images
    """
    
    def __init__(
        self,
        log_transform: bool = True,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (8, 8)
    ):
        """
        Initialize normalizer
        
        Args:
            log_transform: Apply logarithmic transformation
            clahe_enabled: Apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clipping limit
            clahe_tile_grid_size: CLAHE tile grid size (height, width)
        """
        self.log_transform = log_transform
        self.clahe_enabled = clahe_enabled
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        logger.info(f"Normalizer initialized: log={log_transform}, CLAHE={clahe_enabled}")
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply normalization pipeline to sonar image
        
        Args:
            image: 2D sonar image (raw intensity values)
            
        Returns:
            normalized_image: Normalized image in [0, 1] range
        """
        # Ensure image is float
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1] first
        if image.max() > 1.0:
            image = image / image.max()
        
        # Logarithmic transformation
        if self.log_transform:
            image = np.log1p(image)  # log(1 + x) to avoid log(0)
            # Renormalize after log transform
            image = image / image.max()
            logger.debug("Applied logarithmic transformation")
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.clahe_enabled:
            # Convert to uint8 for CLAHE
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Apply CLAHE
            clahe_result = exposure.equalize_adapthist(
                image_uint8,
                clip_limit=self.clahe_clip_limit,
                kernel_size=self.clahe_tile_grid_size
            )
            
            # Handle case where equalize_adapthist might return tuple (shouldn't happen, but be safe)
            if isinstance(clahe_result, tuple):
                image_uint8 = clahe_result[0]  # Take first element if tuple
                logger.warning("equalize_adapthist returned tuple, using first element")
            else:
                image_uint8 = clahe_result
            
            # Convert back to float [0, 1]
            image = image_uint8.astype(np.float32)
            logger.debug("Applied CLAHE contrast enhancement")
        
        # Final normalization to [0, 1]
        image = np.clip(image, 0, 1)
        
        return image

