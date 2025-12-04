"""
Bi-directional Feature Pyramid Network (BiFPN)

Purpose: Weighted bi-directional flow of information between feature levels
Enables effective merging of high-level semantic features with low-level edge details
"""
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer
    """
    
    def __init__(self, channels: int):
        """
        Initialize BiFPN layer
        
        Args:
            channels: Number of channels
        """
        super(BiFPNLayer, self).__init__()
        self.channels = channels
        
        # Convolutions for feature refinement
        self.conv_p3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_p4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_p5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        # Weighted fusion (learnable weights)
        self.weights_p3 = nn.Parameter(torch.ones(2))
        self.weights_p4 = nn.Parameter(torch.ones(3))
        self.weights_p5 = nn.Parameter(torch.ones(2))
        
        # Normalization
        self.eps = 1e-4
        
        logger.debug(f"BiFPNLayer initialized: channels={channels}")
    
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> tuple:
        """
        Forward pass with bi-directional feature fusion
        
        Args:
            p3: Feature map at level 3 (highest resolution)
            p4: Feature map at level 4
            p5: Feature map at level 5 (lowest resolution)
            
        Returns:
            Updated p3, p4, p5 feature maps
        """
        # Top-down pathway (P5 -> P4 -> P3)
        # P5 to P4
        p5_up = nn.functional.interpolate(p5, size=p4.shape[2:], mode='nearest')
        w = torch.softmax(self.weights_p5, dim=0)
        p4_td = w[0] * p4 + w[1] * p5_up
        p4_td = self.conv_p4(p4_td)
        
        # P4 to P3
        p4_up = nn.functional.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        w = torch.softmax(self.weights_p4, dim=0)
        p3_td = w[0] * p3 + w[1] * p4_up
        p3_td = self.conv_p3(p3_td)
        
        # Bottom-up pathway (P3 -> P4 -> P5)
        # P3 to P4
        p3_down = nn.functional.adaptive_avg_pool2d(p3_td, output_size=p4.shape[2:])
        w = torch.softmax(self.weights_p4, dim=0)
        p4_bu = w[0] * p4 + w[1] * p3_down + w[2] * p5
        p4_bu = self.conv_p4(p4_bu)
        
        # P4 to P5
        p4_down = nn.functional.adaptive_avg_pool2d(p4_bu, output_size=p5.shape[2:])
        w = torch.softmax(self.weights_p5, dim=0)
        p5_bu = w[0] * p5 + w[1] * p4_down
        p5_bu = self.conv_p5(p5_bu)
        
        return p3_td, p4_bu, p5_bu


class BiFPN(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    
    def __init__(self, channels: int, num_layers: int = 3):
        """
        Initialize BiFPN
        
        Args:
            channels: Number of channels
            num_layers: Number of BiFPN layers to stack
        """
        super(BiFPN, self).__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            BiFPNLayer(channels) for _ in range(num_layers)
        ])
        
        logger.info(f"BiFPN initialized: channels={channels}, layers={num_layers}")
    
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> tuple:
        """
        Forward pass through multiple BiFPN layers
        
        Args:
            p3, p4, p5: Input feature maps at different scales
            
        Returns:
            Enhanced feature maps
        """
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        
        return p3, p4, p5

