"""
Nadir Masking: Physics-Guided Loss Masking

Purpose: Zero out loss for detections in the water column (blind zone)
Prevents model from learning noise patterns in physically impossible regions
"""
import torch
import numpy as np
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class NadirMasker:
    """
    Applies nadir (blind zone) masking during training
    """
    
    def __init__(self):
        """Initialize nadir masker"""
        logger.debug("NadirMasker initialized")
    
    def create_mask_from_bottom_line(
        self,
        bottom_line: np.ndarray,
        image_shape: tuple,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Create binary mask from bottom line
        
        Args:
            bottom_line: Seabed row indices for each column [width]
            image_shape: (height, width) of image
            device: Device for tensor ('cpu' or 'cuda')
            
        Returns:
            Binary mask [height, width] where True = water column (masked)
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=bool)
        
        for col in range(width):
            if col < len(bottom_line):
                # Everything above bottom_line is water column
                mask[:int(bottom_line[col]), col] = True
        
        return torch.from_numpy(mask).to(device)
    
    def apply_mask_to_loss(
        self,
        loss_map: torch.Tensor,
        blind_zone_mask: torch.Tensor,
        boxes: torch.Tensor,
        image_shape: tuple
    ) -> torch.Tensor:
        """
        Apply nadir masking to loss map
        
        Args:
            loss_map: Loss values [N] or [H, W]
            blind_zone_mask: Binary mask [H, W] (True = water column)
            boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
            image_shape: (height, width) of image
            
        Returns:
            Masked loss map
        """
        if len(loss_map.shape) == 1:
            # Loss per box: check if box center is in blind zone
            height, width = image_shape
            masked_loss = loss_map.clone()
            
            for i, box in enumerate(boxes):
                # Get box center
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                
                # Check if center is in blind zone
                if 0 <= cy < height and 0 <= cx < width:
                    if blind_zone_mask[cy, cx]:
                        masked_loss[i] = 0.0  # Zero out loss in blind zone
            
            return masked_loss
        else:
            # Loss map: directly apply mask
            return loss_map * (~blind_zone_mask).float()
    
    def filter_boxes_in_blind_zone(
        self,
        boxes: torch.Tensor,
        blind_zone_mask: torch.Tensor,
        image_shape: tuple
    ) -> torch.Tensor:
        """
        Filter out boxes that are entirely in the blind zone
        
        Args:
            boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
            blind_zone_mask: Binary mask [H, W]
            image_shape: (height, width)
            
        Returns:
            Boolean mask [N] indicating valid boxes (False = in blind zone)
        """
        height, width = image_shape
        valid_mask = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int()
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Check if box center is in blind zone
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            if 0 <= cy < height and 0 <= cx < width:
                if blind_zone_mask[cy, cx]:
                    valid_mask[i] = False
        
        return valid_mask

