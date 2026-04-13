# architecture_configs.py

# 1. Standard YOLO11m (Baseline)
# We just use "yolo11m.pt" or "yolo11m.yaml" from ultralytics

# 2. SPDConv Architecture (Based on yolo11-sonar.yaml)
SPDCONV_YOLO_YAML = """
scales:
  m: [0.50, 1.00, 512] 

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, SonarSPDConv, [256]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]        # 2
  - [-1, 1, SonarSPDConv, [512]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]        # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 10
  - [[-1, 6], 1, Concat, [1]]  # 11
  - [-1, 2, C3k2, [512, False]] # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 13
  - [[-1, 4], 1, Concat, [1]]  # 14
  - [-1, 2, C3k2, [256, False]] # 15

  - [-1, 1, Conv, [256, 3, 2]] # 16
  - [[-1, 12], 1, Concat, [1]] # 17
  - [-1, 2, C3k2, [512, False]] # 18

  - [-1, 1, Conv, [512, 3, 2]] # 19
  - [[-1, 9], 1, Concat, [1]]  # 20
  - [-1, 2, C3k2, [1024, True]] # 21

  - [[15, 18, 21], 1, Detect, [nc]] # 22
"""

# 3. Attention + SPDConv (Full Hybrid)
ATTN_SPDCONV_YOLO_YAML = """
scales:
  m: [0.50, 1.00, 512] 

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, SonarSPDConv, [256]]       # 1-P2/4
  - [-1, 2, C3k2, [128, False]]        # 2
  - [-1, 1, SonarSPDConv, [512]]       # 3-P3/8
  - [-1, 2, C3k2, [256, False]]        # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, CBAM, [512]]               # 7 - Attention
  - [-1, 1, Conv, [1024, 3, 2]]        # 8-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 9
  - [-1, 1, SPPF, [1024, 5]]           # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 11
  - [[-1, 6], 1, BiFPN_Concat2, [512]] # 12 - BiFPN Fusion
  - [-1, 2, C3k2, [512, False]]        # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 14
  - [[-1, 4], 1, BiFPN_Concat2, [256]] # 15 - BiFPN Fusion
  - [-1, 2, C3k2, [256, False]]        # 16

  - [-1, 1, Conv, [256, 3, 2]]         # 17
  - [[-1, 13], 1, BiFPN_Concat2, [512]] # 18 - BiFPN Fusion
  - [-1, 2, C3k2, [512, False]]        # 19

  - [-1, 1, Conv, [512, 3, 2]]         # 20
  - [[-1, 10], 1, BiFPN_Concat2, [1024]] # 21 - BiFPN Fusion
  - [-1, 2, C3k2, [1024, True]]        # 22
  - [-1, 1, CoordAtt, [1024, 1024]]    # 23 - Neck Attention

  - [[16, 19, 22], 1, Detect, [nc]]    # 24
"""
