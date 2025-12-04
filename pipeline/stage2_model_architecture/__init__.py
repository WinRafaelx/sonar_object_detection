"""
Stage 2: YOLO-Sonar Model Architecture
- SPD-Conv (Space-to-Depth Convolution)
- EMA (Efficient Multi-Scale Attention)
- BiFPN (Bi-directional Feature Pyramid Network)
- P2 Detection Head
"""

from .spd_conv import SPDConv
from .ema_attention import EMA
from .bifpn import BiFPN
from .yolo_sonar import YOLOSonar

__all__ = ['SPDConv', 'EMA', 'BiFPN', 'YOLOSonar']

