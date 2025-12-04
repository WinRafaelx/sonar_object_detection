"""
Efficient Multi-Scale Attention (EMA)

Purpose: Aggregate pixel-level attention across parallel subnetworks
Links high-intensity object returns with corresponding low-intensity shadows
"""
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention module
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initialize EMA module
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction factor
        """
        super(EMA, self).__init__()
        self.channels = channels
        
        # Group convolution for multi-scale feature extraction
        self.groups = [1, 2, 4, 8]  # Different group sizes for multi-scale
        
        # Parallel subnetworks
        self.convs = nn.ModuleList()
        for group in self.groups:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels // reduction, kernel_size=1, groups=group),
                    nn.BatchNorm2d(channels // reduction),
                    nn.SiLU()
                )
            )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels // reduction * len(self.groups), channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        logger.debug(f"EMA initialized: channels={channels}, reduction={reduction}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.convs:
            multi_scale_features.append(conv(x))
        
        # Concatenate multi-scale features
        fused = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion(fused)
        
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_sa = x_ca * sa
        
        # Combine with fused features
        out = x_sa + fused
        
        return out

