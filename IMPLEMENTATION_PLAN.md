# Implementation Plan: Hybrid Acoustic-Vision Pipeline

## Project Structure

```
sonar_object_detection/
├── pipeline/
│   ├── __init__.py
│   ├── stage1_signal_conditioning/
│   │   ├── __init__.py
│   │   ├── viterbi_bottom_tracking.py    # Auto-bottom tracking
│   │   ├── wavelet_denoising.py          # 2D DWT denoising
│   │   └── normalization.py              # Log transform + CLAHE
│   ├── stage2_model_architecture/
│   │   ├── __init__.py
│   │   ├── spd_conv.py                   # Space-to-Depth Convolution
│   │   ├── ema_attention.py              # Efficient Multi-Scale Attention
│   │   ├── bifpn.py                      # Bi-directional Feature Pyramid
│   │   └── yolo_sonar.py                 # Main YOLO-Sonar model
│   ├── stage3_training/
│   │   ├── __init__.py
│   │   ├── shape_iou_loss.py             # Shape-IoU loss function
│   │   └── nadir_masking.py              # Physics-guided loss masking
│   ├── stage4_inference/
│   │   ├── __init__.py
│   │   ├── sahi_slicing.py               # SAHI tiling
│   │   ├── weighted_box_fusion.py        # WBF merging
│   │   └── geocoding.py                  # Geometric coordinate conversion
│   └── pipeline_orchestrator.py          # Main pipeline controller
├── config/
│   └── pipeline_config.yaml              # Configuration parameters
├── utils/
│   ├── __init__.py
│   ├── logger.py                          # Logging utilities
│   └── visualization.py                   # Debug visualization tools
├── tests/
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_stage4.py
├── train.py                               # Updated training script
├── inference.py                           # Inference pipeline
└── requirements.txt                       # Updated dependencies
```

## Implementation Order

### Phase 1: Foundation & Stage 1 (Signal Conditioning)
1. Create project structure
2. Set up logging and configuration system
3. Implement Viterbi bottom tracking
4. Implement Wavelet denoising
5. Implement Dynamic range normalization
6. Create Stage 1 integration test

### Phase 2: Stage 2 (Model Architecture)
1. Implement SPD-Conv module
2. Implement EMA attention module
3. Implement BiFPN module
4. Integrate into YOLO-Sonar model
5. Enable P2 detection head
6. Create Stage 2 integration test

### Phase 3: Stage 3 (Training Protocol)
1. Implement Shape-IoU loss function
2. Implement Nadir masking for training
3. Integrate with Ultralytics training loop
4. Create Stage 3 integration test

### Phase 4: Stage 4 (Inference & Geocoding)
1. Implement SAHI slicing
2. Implement Weighted Box Fusion
3. Implement Geometric geocoding
4. Create Stage 4 integration test

### Phase 5: Integration
1. Create pipeline orchestrator
2. End-to-end testing
3. Performance optimization
4. Documentation

## Key Design Decisions

1. **Modularity**: Each stage is independent and can be tested separately
2. **Logging**: Comprehensive logging at each step for error tracking
3. **Configuration**: All parameters in YAML for easy tuning
4. **Backward Compatibility**: Support both original YOLO and YOLO-Sonar
5. **Visualization**: Debug tools to visualize each stage's output

## Dependencies to Add

- PyWavelets (for DWT)
- scikit-image (for CLAHE)
- sahi (for inference slicing)
- ensemble-boxes (for WBF)
- matplotlib (for visualization)
- pyyaml (for configuration)

