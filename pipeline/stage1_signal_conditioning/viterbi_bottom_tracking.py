"""
Auto-Bottom Tracking using Viterbi Algorithm

Purpose:
1. Calculate sensor altitude (H) above seabed for geocoding
2. Generate blind zone mask to suppress water column false positives
"""
import numpy as np
from typing import Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ViterbiBottomTracker:
    """
    Implements Viterbi algorithm for optimal seabed line detection
    """
    
    def __init__(self, max_jump: int = 10, intensity_threshold: float = 0.3):
        """
        Initialize Viterbi bottom tracker
        
        Args:
            max_jump: Maximum vertical jump in pixels between consecutive pings
            intensity_threshold: Minimum intensity for seabed detection
        """
        self.max_jump = max_jump
        self.intensity_threshold = intensity_threshold
        logger.info(f"ViterbiBottomTracker initialized: max_jump={max_jump}, threshold={intensity_threshold}")
    
    def detect_bottom_line(self, sonar_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect seabed line using Viterbi algorithm
        
        Args:
            sonar_image: 2D sonar image (height x width), normalized to [0, 1]
            
        Returns:
            bottom_line: Array of seabed row indices for each column (ping)
            blind_zone_mask: Binary mask (True = water column, False = seabed region)
        """
        height, width = sonar_image.shape
        logger.debug(f"Processing sonar image: {height}x{width}")
        
        # Initialize cost matrix (negative log probability)
        # Higher intensity = lower cost (more likely to be seabed)
        cost_matrix = -np.log(sonar_image + 1e-10)
        
        # Initialize DP table: dp[i, j] = minimum cost to reach row i at column j
        dp = np.full((height, width), np.inf)
        backpointer = np.zeros((height, width), dtype=int)
        
        # Initialize first column: cost is just the intensity cost
        dp[:, 0] = cost_matrix[:, 0]
        
        # Forward pass: fill DP table
        for j in range(1, width):
            for i in range(height):
                # Consider transitions from previous column
                # Allow transitions within max_jump pixels
                start_row = max(0, i - self.max_jump)
                end_row = min(height, i + self.max_jump + 1)
                
                # Find minimum cost transition
                prev_costs = dp[start_row:end_row, j-1]
                transition_costs = np.abs(np.arange(start_row, end_row) - i)
                
                total_costs = prev_costs + transition_costs + cost_matrix[i, j]
                min_idx = np.argmin(total_costs)
                min_cost = total_costs[min_idx]
                
                dp[i, j] = min_cost
                backpointer[i, j] = start_row + min_idx
        
        # Backward pass: trace optimal path
        bottom_line = np.zeros(width, dtype=int)
        bottom_line[width - 1] = np.argmin(dp[:, width - 1])
        
        for j in range(width - 2, -1, -1):
            bottom_line[j] = backpointer[bottom_line[j + 1], j + 1]
        
        # Generate blind zone mask (water column = True)
        blind_zone_mask = np.zeros((height, width), dtype=bool)
        for j in range(width):
            blind_zone_mask[:bottom_line[j], j] = True
        
        # Calculate altitude (height above seabed) for each ping
        altitude = height - bottom_line
        
        logger.info(f"Bottom line detected. Mean altitude: {np.mean(altitude):.2f} pixels")
        logger.debug(f"Bottom line range: {np.min(bottom_line)} - {np.max(bottom_line)}")
        
        return bottom_line, blind_zone_mask
    
    def get_altitude(self, bottom_line: np.ndarray, image_height: int) -> np.ndarray:
        """
        Calculate sensor altitude above seabed for each ping
        
        Args:
            bottom_line: Seabed row indices
            image_height: Total image height in pixels
            
        Returns:
            altitude: Array of altitude values (in pixels) for each ping
        """
        altitude = image_height - bottom_line
        return altitude

