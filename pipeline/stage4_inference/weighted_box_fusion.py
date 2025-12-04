"""
Weighted Box Fusion (WBF)

Purpose: Merge overlapping detections from adjacent tiles
Averages coordinates based on confidence scores for increased precision
"""
import numpy as np
from typing import List, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WBF:
    """
    Weighted Box Fusion algorithm
    """
    
    def __init__(self, iou_threshold: float = 0.5, skip_box_threshold: float = 0.0001):
        """
        Initialize WBF
        
        Args:
            iou_threshold: IoU threshold for box clustering
            skip_box_threshold: Minimum confidence to consider a box
        """
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        logger.info(f"WBF initialized: iou_threshold={iou_threshold}")
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes
        
        Args:
            box1: Box [x1, y1, x2, y2]
            box2: Box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def fuse_boxes(self, boxes: List[np.ndarray], confidences: List[float]) -> np.ndarray:
        """
        Fuse boxes using weighted average
        
        Args:
            boxes: List of boxes [x1, y1, x2, y2]
            confidences: List of confidence scores
            
        Returns:
            Fused box [x1, y1, x2, y2]
        """
        confidences = np.array(confidences)
        weights = confidences / (confidences.sum() + 1e-7)
        
        boxes = np.array(boxes)
        fused = np.zeros(4)
        
        for i in range(4):
            fused[i] = np.average(boxes[:, i], weights=weights)
        
        return fused
    
    def merge(self, detections: np.ndarray) -> np.ndarray:
        """
        Merge detections using WBF
        
        Args:
            detections: Detection array [N, 6] (x1, y1, x2, y2, conf, cls)
            
        Returns:
            Merged detections [M, 6]
        """
        if len(detections) == 0:
            return detections
        
        # Filter by confidence threshold
        valid_mask = detections[:, 4] >= self.skip_box_threshold
        detections = detections[valid_mask]
        
        if len(detections) == 0:
            return np.array([]).reshape(0, 6)
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(-detections[:, 4])
        detections = detections[sorted_indices]
        
        merged = []
        used = np.zeros(len(detections), dtype=bool)
        
        for i in range(len(detections)):
            if used[i]:
                continue
            
            # Find boxes that overlap with this box
            cluster_indices = [i]
            cluster_boxes = [detections[i, :4]]
            cluster_confs = [detections[i, 4]]
            cluster_cls = detections[i, 5]
            
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                
                # Check same class
                if detections[j, 5] != cluster_cls:
                    continue
                
                # Check IoU
                iou = self.compute_iou(detections[i, :4], detections[j, :4])
                if iou >= self.iou_threshold:
                    cluster_indices.append(j)
                    cluster_boxes.append(detections[j, :4])
                    cluster_confs.append(detections[j, 4])
                    used[j] = True
            
            # Fuse boxes in cluster
            fused_box = self.fuse_boxes(cluster_boxes, cluster_confs)
            
            # Average confidence
            avg_conf = np.mean(cluster_confs)
            
            merged.append([*fused_box, avg_conf, cluster_cls])
            used[i] = True
        
        if merged:
            result = np.array(merged)
            logger.debug(f"WBF: {len(detections)} -> {len(result)} detections")
            return result
        else:
            return np.array([]).reshape(0, 6)

