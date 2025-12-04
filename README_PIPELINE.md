# Hybrid Acoustic-Vision Pipeline for Sonar Object Detection

This implementation provides a complete, step-by-step pipeline for side-scan sonar object detection as described in your system design document. Each stage can be tested and debugged independently.

## Project Structure

```
sonar_object_detection/
├── pipeline/
│   ├── stage1_signal_conditioning/    # Physics-based preprocessing
│   │   ├── viterbi_bottom_tracking.py
│   │   ├── wavelet_denoising.py
│   │   ├── normalization.py
│   │   └── processor.py
│   ├── stage2_model_architecture/     # YOLO-Sonar model
│   │   ├── spd_conv.py
│   │   ├── ema_attention.py
│   │   ├── bifpn.py
│   │   └── yolo_sonar.py
│   ├── stage3_training/               # Training protocol
│   │   ├── shape_iou_loss.py
│   │   └── nadir_masking.py
│   ├── stage4_inference/             # Inference & geocoding
│   │   ├── sahi_slicing.py
│   │   ├── weighted_box_fusion.py
│   │   └── geocoding.py
│   └── pipeline_orchestrator.py      # Main coordinator
├── config/
│   └── pipeline_config.yaml          # Configuration parameters
├── utils/
│   ├── logger.py                     # Logging utilities
│   └── visualization.py              # Debug visualization
├── train_sonar.py                    # Training script
├── inference.py                      # Inference script
└── test_pipeline.py                  # Stage-by-stage testing
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a trained YOLO model or use the base YOLOv11 model.

## Usage

### Testing Individual Stages

Test Stage 1 (Signal Conditioning) on a sample image:
```bash
python test_pipeline.py --image path/to/sonar_image.png --stage 1
```

This will:
- Run Viterbi bottom tracking
- Apply wavelet denoising
- Apply normalization
- Save visualization to `test_output/stage1_test.png`

### Training

Train the YOLO-Sonar model:
```bash
python train_sonar.py --data sonar_detect-1/data.yaml --model yolo11m.pt --epochs 50
```

**Note:** Full integration of Shape-IoU loss and Nadir masking requires custom training loop modifications to Ultralytics. The current implementation provides the framework; full integration would require:
1. Applying Stage 1 preprocessing to training images
2. Replacing standard IoU loss with ShapeIoULoss
3. Applying NadirMasker during loss calculation

### Inference

Run complete pipeline on a sonar image:
```bash
python inference.py --image path/to/sonar_image.png --model runs/train/yolo_sonar/weights/best.pt --visualize
```

This will:
1. **Stage 1:** Preprocess the image (bottom tracking, denoising, normalization)
2. **Stage 2:** Run model inference (with SAHI slicing if enabled)
3. **Stage 4:** Merge detections (WBF) and geocode to real-world coordinates
4. Save results and visualizations to the output directory

### Configuration

Edit `config/pipeline_config.yaml` to adjust:
- Viterbi parameters (max_jump, intensity_threshold)
- Wavelet denoising (wavelet type, threshold)
- Normalization (CLAHE parameters)
- Model architecture settings
- SAHI slicing parameters
- WBF parameters
- Geocoding parameters

## Pipeline Stages

### Stage 1: Physics-Based Signal Conditioning

**Components:**
- **Viterbi Bottom Tracking:** Detects seabed line using dynamic programming
  - Outputs: `bottom_line`, `blind_zone_mask`, `altitude`
- **Wavelet Denoising:** Removes speckle noise while preserving edges
  - Uses 2D DWT with soft thresholding
- **Normalization:** Log transform + CLAHE for dynamic range compression

**Usage:**
```python
from pipeline.stage1_signal_conditioning.processor import Stage1Processor
import yaml

with open('config/pipeline_config.yaml') as f:
    config = yaml.safe_load(f)

processor = Stage1Processor(config)
processed_image, metadata = processor.process(sonar_image)
```

### Stage 2: YOLO-Sonar Architecture

**Modifications:**
- **SPD-Conv:** Replaces strided convolutions to preserve fine features
- **EMA Attention:** Multi-scale attention for linking object returns with shadows
- **BiFPN:** Bi-directional feature pyramid for multi-scale fusion
- **P2 Detection Head:** High-resolution detection grid

**Note:** Full integration requires modifying Ultralytics YOLO source code. The current implementation provides the modules and a wrapper framework.

### Stage 3: Training Protocol

**Components:**
- **Shape-IoU Loss:** Penalizes aspect ratio deviations for rigid geometries
- **Nadir Masking:** Zeros out loss in water column (blind zone)

**Usage:**
```python
from pipeline.stage3_training.shape_iou_loss import ShapeIoULoss
from pipeline.stage3_training.nadir_masking import NadirMasker

loss_fn = ShapeIoULoss(weight=1.0)
masker = NadirMasker()
```

### Stage 4: Inference & Geocoding

**Components:**
- **SAHI Slicing:** Divides high-res images into overlapping tiles
- **WBF:** Merges overlapping detections with confidence weighting
- **Geocoding:** Converts pixels to real-world coordinates using slant range correction

**Usage:**
```python
from pipeline.stage4_inference.sahi_slicing import SAHISlicer
from pipeline.stage4_inference.weighted_box_fusion import WBF
from pipeline.stage4_inference.geocoding import Geocoder

slicer = SAHISlicer(slice_size=640, overlap_ratio=0.25)
wbf = WBF(iou_threshold=0.5)
geocoder = Geocoder(sonar_range=50.0, pixel_resolution=0.1)
```

## Logging

All stages log their operations to:
- Console (INFO level)
- `logs/pipeline_YYYYMMDD_HHMMSS.log` (DEBUG level)

Use logging to track:
- Which stage is executing
- Processing times
- Intermediate results
- Errors and warnings

## Error Tracking

Each stage is designed to:
1. Log detailed information at each step
2. Return intermediate results for debugging
3. Handle errors gracefully with informative messages

To debug a specific stage:
1. Run `test_pipeline.py` with the specific stage
2. Check the log file for detailed execution traces
3. Use visualization tools to inspect intermediate outputs

## Next Steps

1. **Full YOLO Integration:** Modify Ultralytics source to integrate SPD-Conv, EMA, and BiFPN
2. **Custom Training Loop:** Implement full Shape-IoU and Nadir masking in training
3. **Performance Optimization:** Optimize each stage for production use
4. **Validation:** Test on your sonar dataset and tune parameters

## Citation

If you use this implementation, please cite the papers referenced in your system design document for:
- Wavelet denoising [cite: 800]
- SPD-Conv [cite: 33]
- EMA attention [cite: 799]
- BiFPN [cite: 799]
- Shape-IoU [cite: 799]

