"""
Stage 3: Specialized Training Protocol
- Shape-IoU Loss Function
- Nadir Masking (Physics-Guided Loss)
"""

from .shape_iou_loss import ShapeIoULoss
from .nadir_masking import NadirMasker

__all__ = ['ShapeIoULoss', 'NadirMasker']

