import torch
import torch.nn as nn

# 1. SonarSPDConv: Space-to-Depth Downsampler compatible with YOLO scaling
class SonarSPDConv(nn.Module):
    def __init__(self, c1, c2, k=1, dimension=1):
        super().__init__()
        self.d = dimension
        # SPD-Conv stacks 4 sub-pixels into the channel dimension
        # We add a 1x1 convolution to transition from c1*4 to the desired c2
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2) if (k > 0 or c2 != c1 * 4) else nn.Identity()

    def forward(self, x):
        # Slice and stack sub-regions: [N, C, H, W] -> [N, C*4, H/2, W/2]
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], 
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=self.d)
        return self.conv(x)

# 2. CoordAtt: Coordinate Attention compatible with YOLO scaling
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()
