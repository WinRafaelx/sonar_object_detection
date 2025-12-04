"""
SAHI (Slicing Aided Hyper Inference) Slicing

Purpose: Slice high-resolution sonar waterfall into overlapping tiles
Maintains resolution for small object detection
"""
import numpy as np
from typing import List, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SAHISlicer:
    """
    Implements SAHI-style image slicing with overlap
    """
    
    def __init__(self, slice_size: int = 640, overlap_ratio: float = 0.25):
        """
        Initialize SAHI slicer
        
        Args:
            slice_size: Size of each slice (square)
            overlap_ratio: Overlap ratio between adjacent slices (0.0 to 1.0)
        """
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(slice_size * (1 - overlap_ratio))
        
        logger.info(f"SAHISlicer initialized: slice_size={slice_size}, overlap={overlap_ratio}")
    
    def slice_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Slice image into overlapping tiles
        
        Args:
            image: Input image [H, W] or [H, W, C]
            
        Returns:
            List of (slice, (y_offset, x_offset)) tuples
        """
        height, width = image.shape[:2]
        slices = []
        
        # Calculate number of slices
        y_steps = (height - self.slice_size) // self.stride + 1
        x_steps = (width - self.slice_size) // self.stride + 1
        
        logger.debug(f"Slicing image {height}x{width} into {y_steps}x{x_steps} tiles")
        
        for y in range(0, height - self.slice_size + 1, self.stride):
            for x in range(0, width - self.slice_size + 1, self.stride):
                # Extract slice
                if len(image.shape) == 2:
                    slice_img = image[y:y+self.slice_size, x:x+self.slice_size]
                else:
                    slice_img = image[y:y+self.slice_size, x:x+self.slice_size, :]
                
                slices.append((slice_img, (y, x)))
        
        # Handle edge cases: ensure full coverage
        # Right edge
        if width % self.stride != 0:
            x = width - self.slice_size
            for y in range(0, height - self.slice_size + 1, self.stride):
                if len(image.shape) == 2:
                    slice_img = image[y:y+self.slice_size, x:x+self.slice_size]
                else:
                    slice_img = image[y:y+self.slice_size, x:x+self.slice_size, :]
                slices.append((slice_img, (y, x)))
        
        # Bottom edge
        if height % self.stride != 0:
            y = height - self.slice_size
            for x in range(0, width - self.slice_size + 1, self.stride):
                if len(image.shape) == 2:
                    slice_img = image[y:y+self.slice_size, x:x+self.slice_size]
                else:
                    slice_img = image[y:y+self.slice_size, x:x+self.slice_size, :]
                slices.append((slice_img, (y, x)))
        
        # Bottom-right corner
        if height % self.stride != 0 and width % self.stride != 0:
            y = height - self.slice_size
            x = width - self.slice_size
            if len(image.shape) == 2:
                slice_img = image[y:y+self.slice_size, x:x+self.slice_size]
            else:
                slice_img = image[y:y+self.slice_size, x:x+self.slice_size, :]
            slices.append((slice_img, (y, x)))
        
        logger.info(f"Generated {len(slices)} slices")
        return slices
    
    def merge_detections(
        self,
        detections: List[np.ndarray],
        offsets: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Merge detections from slices back to original image coordinates
        
        Args:
            detections: List of detection arrays [N, 6] (x1, y1, x2, y2, conf, cls)
            offsets: List of (y_offset, x_offset) for each slice
            
        Returns:
            Merged detections in original image coordinates
        """
        merged = []
        
        for dets, (y_off, x_off) in zip(detections, offsets):
            if len(dets) > 0:
                # Adjust coordinates
                dets_adjusted = dets.copy()
                dets_adjusted[:, [0, 2]] += x_off  # x coordinates
                dets_adjusted[:, [1, 3]] += y_off  # y coordinates
                merged.append(dets_adjusted)
        
        if merged:
            return np.vstack(merged)
        else:
            return np.array([]).reshape(0, 6)

