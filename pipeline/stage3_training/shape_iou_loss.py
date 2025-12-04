"""
Shape-IoU Loss Function

Purpose: Penalize aspect ratio deviations to recognize rigid geometric properties
of man-made threats (cylinders) vs irregular rocks
"""
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ShapeIoULoss(nn.Module):
    """
    Shape-IoU Loss: Focuses on aspect ratio matching for rigid geometries
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize Shape-IoU loss
        
        Args:
            weight: Loss weight
        """
        super(ShapeIoULoss, self).__init__()
        self.weight = weight
        logger.debug(f"ShapeIoULoss initialized: weight={weight}")
    
    def bbox_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
        
        Args:
            boxes: Boxes in center format [N, 4]
            
        Returns:
            Boxes in corner format [N, 4]
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between boxes
        
        Args:
            box1: Boxes 1 [N, 4] in (x1, y1, x2, y2) format
            box2: Boxes 2 [N, 4] in (x1, y1, x2, y2) format
            
        Returns:
            IoU values [N]
        """
        # Calculate intersection
        x1_inter = torch.max(box1[:, 0], box2[:, 0])
        y1_inter = torch.max(box1[:, 1], box2[:, 1])
        x2_inter = torch.min(box1[:, 2], box2[:, 2])
        y2_inter = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
        
        # Calculate union
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-7)
        return iou
    
    def compute_aspect_ratio_penalty(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute aspect ratio penalty
        
        Args:
            pred_boxes: Predicted boxes [N, 4] in (cx, cy, w, h) format
            target_boxes: Target boxes [N, 4] in (cx, cy, w, h) format
            
        Returns:
            Aspect ratio penalty [N]
        """
        # Calculate aspect ratios (w/h)
        pred_ar = pred_boxes[:, 2] / (pred_boxes[:, 3] + 1e-7)
        target_ar = target_boxes[:, 2] / (target_boxes[:, 3] + 1e-7)
        
        # Penalty based on aspect ratio difference
        ar_diff = torch.abs(pred_ar - target_ar) / (target_ar + 1e-7)
        penalty = torch.exp(-ar_diff)  # Higher penalty for larger differences
        
        return penalty
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Shape-IoU loss
        
        Args:
            pred_boxes: Predicted boxes [N, 4] in (cx, cy, w, h) format
            target_boxes: Target boxes [N, 4] in (cx, cy, w, h) format
            
        Returns:
            Loss value
        """
        # Convert to corner format for IoU calculation
        pred_xyxy = self.bbox_to_xyxy(pred_boxes)
        target_xyxy = self.bbox_to_xyxy(target_boxes)
        
        # Compute standard IoU
        iou = self.compute_iou(pred_xyxy, target_xyxy)
        
        # Compute aspect ratio penalty
        ar_penalty = self.compute_aspect_ratio_penalty(pred_boxes, target_boxes)
        
        # Shape-IoU: Combine IoU with aspect ratio penalty
        # Lower penalty = higher weight on IoU
        shape_iou = iou * ar_penalty
        
        # Loss is 1 - Shape-IoU
        loss = (1 - shape_iou).mean()
        
        return loss * self.weight

