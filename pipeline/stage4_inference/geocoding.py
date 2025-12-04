"""
Geometric Geocoding

Purpose: Convert pixel coordinates to real-world coordinates using slant range correction
"""
import numpy as np
from typing import Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Geocoder:
    """
    Converts pixel detections to real-world geographic coordinates
    """
    
    def __init__(
        self,
        sonar_range: float = 50.0,
        pixel_resolution: float = 0.1,
        image_height: Optional[int] = None
    ):
        """
        Initialize geocoder
        
        Args:
            sonar_range: Maximum sonar range in meters
            pixel_resolution: Meters per pixel
            image_height: Image height in pixels (for altitude calculation)
        """
        self.sonar_range = sonar_range
        self.pixel_resolution = pixel_resolution
        self.image_height = image_height
        logger.info(f"Geocoder initialized: range={sonar_range}m, resolution={pixel_resolution}m/pixel")
    
    def slant_range_to_ground_range(
        self,
        slant_range: float,
        altitude: float
    ) -> float:
        """
        Convert slant range to ground range using slant range correction
        
        Formula: Ground_Range = sqrt(Slant_Range^2 - Altitude^2)
        
        Args:
            slant_range: Slant range in meters
            altitude: Sensor altitude above seabed in meters
            
        Returns:
            Ground range in meters
        """
        if slant_range <= altitude:
            return 0.0
        
        ground_range = np.sqrt(slant_range**2 - altitude**2)
        return ground_range
    
    def pixel_to_slant_range(
        self,
        pixel_y: float,
        image_height: Optional[int] = None
    ) -> float:
        """
        Convert pixel Y coordinate to slant range
        
        Args:
            pixel_y: Pixel Y coordinate (row)
            image_height: Image height (uses self.image_height if None)
            
        Returns:
            Slant range in meters
        """
        if image_height is None:
            image_height = self.image_height
            if image_height is None:
                raise ValueError("image_height must be provided")
        
        # Assuming pixel_y=0 is at top (near sensor), pixel_y=height is at max range
        # Slant range increases linearly with pixel_y
        slant_range = (pixel_y / image_height) * self.sonar_range
        return slant_range
    
    def geocode_detections(
        self,
        detections: np.ndarray,
        altitude: np.ndarray,
        ping_index: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Geocode detections to real-world coordinates
        
        Args:
            detections: Detection array [N, 6] (x1, y1, x2, y2, conf, cls)
            altitude: Altitude array [width] for each ping (column)
            ping_index: Ping indices for each detection [N] (uses x center if None)
            
        Returns:
            Geocoded detections [N, 8] (x1, y1, x2, y2, conf, cls, ground_x, ground_y)
        """
        if len(detections) == 0:
            return np.array([]).reshape(0, 8)
        
        geocoded = np.zeros((len(detections), 8))
        geocoded[:, :6] = detections  # Copy original detections
        
        if self.image_height is None:
            logger.warning("image_height not set, using default calculation")
            # Estimate from detection coordinates
            self.image_height = int(np.max(detections[:, 3])) + 1
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det[:4]
            
            # Get detection center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Get ping index (column)
            if ping_index is not None:
                ping_idx = int(ping_index[i])
            else:
                ping_idx = int(cx)
            
            # Get altitude for this ping
            if ping_idx < len(altitude):
                alt = altitude[ping_idx] * self.pixel_resolution  # Convert to meters
            else:
                alt = np.mean(altitude) * self.pixel_resolution
            
            # Convert pixel coordinates to slant range
            slant_range = self.pixel_to_slant_range(cy, self.image_height)
            
            # Convert to ground range
            ground_range = self.slant_range_to_ground_range(slant_range, alt)
            
            # Store geocoded coordinates
            # X: along-track distance (ping index * resolution)
            # Y: cross-track distance (ground range)
            geocoded[i, 6] = ping_idx * self.pixel_resolution  # Ground X (along-track)
            geocoded[i, 7] = ground_range  # Ground Y (cross-track)
        
        logger.debug(f"Geocoded {len(detections)} detections")
        return geocoded

