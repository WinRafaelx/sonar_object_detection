"""
Space-to-Depth Convolution (SPD-Conv)

Purpose: Replace strided convolutions to preserve fine-grained features for micro-targets
"""
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution module
    
    Maps spatial blocks (e.g., 2x2) into channel dimension instead of discarding them
    """
    
    def __init__(self, in_channels: int, out_channels: int, block_size: int = 2):
        """
        Initialize SPD-Conv module
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            block_size: Spatial block size (typically 2)
        """
        super(SPDConv, self).__init__()
        self.block_size = block_size
        
        # Calculate channels after space-to-depth
        # Each 2x2 block becomes 4 channels
        mid_channels = in_channels * (block_size ** 2)
        
        # Convolution after space-to-depth
        self.conv = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation
        
        logger.debug(f"SPDConv: {in_channels} -> {out_channels}, block_size={block_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, out_channels, H//block_size, W//block_size]
        """
        B, C, H, W = x.shape
        
        # Space-to-depth transformation
        # Split into blocks and rearrange
        block_size = self.block_size
        
        # Ensure dimensions are divisible by block_size
        if H % block_size != 0 or W % block_size != 0:
            # Pad if necessary
            pad_h = (block_size - H % block_size) % block_size
            pad_w = (block_size - W % block_size) % block_size
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]
        
        # Reshape: [B, C, H, W] -> [B, C, H//block_size, block_size, W//block_size, block_size]
        x = x.view(B, C, H // block_size, block_size, W // block_size, block_size)
        
        # Permute and reshape: [B, C*block_size^2, H//block_size, W//block_size]
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * (block_size ** 2), H // block_size, W // block_size)
        
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x

