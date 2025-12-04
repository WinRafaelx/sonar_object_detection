# Pipeline Implementation Summary for Report

## Overview

This document summarizes the implementation of the **Hybrid Acoustic-Vision Pipeline** for side-scan sonar object detection. The pipeline integrates physics-based signal processing with a domain-specialized deep learning model, following the system design outlined in Chapter 3 of the senior project report.

## Architecture Overview

The pipeline is structured as a **four-stage modular system**, where each stage can be independently tested and debugged:

1. **Stage 1: Physics-Based Signal Conditioning**
2. **Stage 2: YOLO-Sonar Model Architecture**
3. **Stage 3: Specialized Training Protocol**
4. **Stage 4: Operational Inference & Geocoding**

## Stage 1: Physics-Based Signal Conditioning

### Implementation Details

**1.1 Auto-Bottom Tracking (Viterbi Algorithm)**
- **File**: `pipeline/stage1_signal_conditioning/viterbi_bottom_tracking.py`
- **Purpose**: Detects the seabed line (water column/seabed boundary) using dynamic programming
- **Algorithm**: 
  - Constructs a cost matrix based on acoustic intensity (higher intensity = lower cost)
  - Uses forward-backward dynamic programming to find optimal path
  - Handles vertical jumps between pings (configurable `max_jump` parameter)
- **Outputs**:
  - `bottom_line`: Array of seabed row indices for each ping (column)
  - `blind_zone_mask`: Binary mask identifying water column region
  - `altitude`: Sensor height above seabed (required for geocoding)
- **Key Implementation Notes**:
  - Handles both 2D and 3D image arrays (squeezes channel dimension if present)
  - Uses logarithmic cost function to emphasize high-intensity seabed returns
  - Implements boundary conditions to prevent path from going out of bounds

**1.2 2D Discrete Wavelet Denoising**
- **File**: `pipeline/stage1_signal_conditioning/wavelet_denoising.py`
- **Purpose**: Removes speckle noise while preserving sharp edges of target shadows
- **Method**:
  - Uses PyWavelets library for 2D DWT decomposition
  - Decomposes image into approximation and detail sub-bands (horizontal, vertical, diagonal)
  - Applies soft thresholding to detail coefficients using universal threshold rule
  - Reconstructs image from thresholded coefficients
- **Parameters**:
  - Wavelet type: Daubechies 4 (`db4`) - configurable
  - Threshold mode: Soft thresholding (preserves edge information)
  - Noise estimation: Median Absolute Deviation (MAD) from detail coefficients
- **Justification**: Wavelet-based preprocessing preserves high-frequency edge details while suppressing noise, critical for detecting small target shadows

**1.3 Dynamic Range Normalization**
- **File**: `pipeline/stage1_signal_conditioning/normalization.py`
- **Purpose**: Standardizes acoustic returns for CNN input
- **Pipeline**:
  1. **Logarithmic Transformation**: `I' = log(I + 1)` to compress high-intensity signals
  2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local contrast enhancement
     - Uses scikit-image's `equalize_adapthist`
     - Tile-based processing (8×8 grid, configurable)
     - Clip limit prevents over-amplification
- **Output**: Normalized image in [0, 1] range suitable for neural network input

**1.4 Integrated Processor**
- **File**: `pipeline/stage1_signal_conditioning/processor.py`
- **Function**: Coordinates all Stage 1 components in sequence
- **Processing Order**:
  1. Input normalization to [0, 1]
  2. Viterbi bottom tracking (if enabled)
  3. Wavelet denoising (if enabled)
  4. Dynamic range normalization (if enabled)
- **Output**: Processed image + metadata dictionary containing bottom_line, blind_zone_mask, altitude

## Stage 2: YOLO-Sonar Model Architecture

### Implementation Details

**2.1 SPD-Conv (Space-to-Depth Convolution)**
- **File**: `pipeline/stage2_model_architecture/spd_conv.py`
- **Purpose**: Replaces strided convolutions to preserve fine-grained features for micro-targets
- **Mechanism**:
  - Maps spatial blocks (e.g., 2×2) into channel dimension instead of discarding pixels
  - Converts `[B, C, H, W]` → `[B, C×4, H/2, W/2]` (for block_size=2)
  - Ensures 100% information retention vs. 75% loss with standard strided convolution
- **Implementation**: PyTorch module with BatchNorm and SiLU activation

**2.2 EMA (Efficient Multi-Scale Attention)**
- **File**: `pipeline/stage2_model_architecture/ema_attention.py`
- **Purpose**: Aggregates pixel-level attention across parallel subnetworks
- **Mechanism**:
  - Parallel group convolutions with different group sizes (1, 2, 4, 8) for multi-scale features
  - Channel attention: Adaptive average pooling + learned weights
  - Spatial attention: Combines average and max pooling
  - Links high-intensity object returns with corresponding low-intensity shadows
- **Implementation**: PyTorch module with learnable attention weights

**2.3 BiFPN (Bi-directional Feature Pyramid Network)**
- **File**: `pipeline/stage2_model_architecture/bifpn.py`
- **Purpose**: Weighted bi-directional flow of information between feature levels
- **Mechanism**:
  - Top-down pathway: High-level semantic features flow to low-level features
  - Bottom-up pathway: Low-level edge details flow to high-level features
  - Learnable fusion weights for each connection
  - Enables effective merging of shape information with edge details
- **Implementation**: Stackable layers with depthwise separable convolutions

**2.4 YOLO-Sonar Wrapper**
- **File**: `pipeline/stage2_model_architecture/yolo_sonar.py`
- **Function**: Integrates custom modules with Ultralytics YOLO framework
- **Note**: Full integration requires modifying Ultralytics source code. Current implementation provides framework for integration.

## Stage 3: Specialized Training Protocol

### Implementation Details

**3.1 Shape-IoU Loss Function**
- **File**: `pipeline/stage3_training/shape_iou_loss.py`
- **Purpose**: Penalizes aspect ratio deviations to recognize rigid geometric properties
- **Mechanism**:
  - Computes standard IoU between predicted and ground truth boxes
  - Calculates aspect ratio penalty: `penalty = exp(-|pred_ar - target_ar| / target_ar)`
  - Combined loss: `Shape-IoU = IoU × penalty`
  - Final loss: `Loss = 1 - Shape-IoU`
- **Justification**: Man-made threats (cylinders) have characteristic aspect ratios that distinguish them from irregular rocks

**3.2 Nadir Masking (Physics-Guided Loss)**
- **File**: `pipeline/stage3_training/nadir_masking.py`
- **Purpose**: Zeros out loss for detections in the water column (blind zone)
- **Mechanism**:
  - Uses blind zone mask from Stage 1 (Viterbi output)
  - Checks if bounding box center is in blind zone
  - Multiplies loss by zero for invalid detections
  - Prevents model from learning noise patterns in physically impossible regions
- **Implementation**: Applied during training loop (requires custom training integration)

## Stage 4: Operational Inference & Geocoding

### Implementation Details

**4.1 SAHI Slicing**
- **File**: `pipeline/stage4_inference/sahi_slicing.py`
- **Purpose**: Slice high-resolution sonar waterfall into overlapping tiles
- **Mechanism**:
  - Divides image into 640×640 pixel tiles with 25% overlap (configurable)
  - Ensures small objects aren't lost at image boundaries
  - Merges detections from adjacent tiles back to original coordinates
- **Parameters**: Slice size, overlap ratio

**4.2 Weighted Box Fusion (WBF)**
- **File**: `pipeline/stage4_inference/weighted_box_fusion.py`
- **Purpose**: Merge overlapping detections from adjacent tiles
- **Mechanism**:
  - Clusters detections by IoU threshold
  - Weighted average of box coordinates based on confidence scores
  - Average confidence for merged boxes
  - Improves precision by combining multiple views of same object
- **Algorithm**: IoU-based clustering + confidence-weighted averaging

**4.3 Geometric Geocoding**
- **File**: `pipeline/stage4_inference/geocoding.py`
- **Purpose**: Convert pixel coordinates to real-world coordinates
- **Method**: Slant Range Correction
  - Formula: `Ground_Range = sqrt(Slant_Range² - Altitude²)`
  - Uses altitude from Stage 1 (Viterbi output)
  - Converts pixel Y-coordinate to slant range
  - Outputs: Along-track distance (X) and cross-track distance (Y) in meters
- **Parameters**: Sonar range, pixel resolution

## Pipeline Orchestration

### Main Components

**Pipeline Orchestrator**
- **File**: `pipeline/pipeline_orchestrator.py`
- **Function**: Coordinates all four stages in sequence
- **Workflow**:
  1. Loads and preprocesses image (Stage 1)
  2. Runs model inference (Stage 2)
  3. Applies SAHI slicing if enabled
  4. Merges detections with WBF
  5. Geocodes to real-world coordinates

**Preprocessing Script**
- **File**: `preprocess_dataset.py`
- **Function**: Batch preprocessing of entire dataset
- **Process**:
  - Processes all train/val/test images through Stage 1
  - Saves preprocessed images to new directory
  - Copies labels (unchanged)
  - Creates new data.yaml with preprocessed paths

**Training Script**
- **File**: `train_sonar_complete.py`
- **Function**: Complete pipeline execution
- **Workflow**:
  1. Downloads or loads dataset
  2. Preprocesses dataset (Stage 1)
  3. Trains model (Stage 2)
  4. Evaluates model (optional)

## Key Implementation Challenges and Solutions

### Challenge 1: Image Dimension Handling
- **Problem**: Images loaded with `cv2.IMREAD_GRAYSCALE` sometimes have shape `(H, W, 1)` instead of `(H, W)`
- **Solution**: Added dimension squeezing in both preprocessing and Viterbi code to handle 2D and 3D arrays

### Challenge 2: Path Resolution
- **Problem**: Data.yaml uses relative paths (`../train/images`) that need proper resolution
- **Solution**: Implemented multi-strategy path resolution that tries:
  1. Path as-is from data.yaml location
  2. Path without `../` prefix (if directories are in same folder)
  3. Direct split name with `images` subdirectory

### Challenge 3: Error Tracking
- **Problem**: Need to track errors at each stage for debugging
- **Solution**: Comprehensive logging system with:
  - Console output (INFO level)
  - File logging (DEBUG level)
  - Error samples (first 3 errors with full tracebacks)
  - Progress tracking per stage

### Challenge 4: Modularity
- **Problem**: Need to test each stage independently
- **Solution**: Created separate test script (`test_pipeline.py`) and individual component modules that can be imported and tested separately

## Configuration Management

**Configuration File**: `config/pipeline_config.yaml`
- Centralized parameter management
- YAML format for easy editing
- Separate sections for each stage
- Enables/disables components per stage

## File Structure

```
sonar_object_detection/
├── pipeline/
│   ├── stage1_signal_conditioning/    # Stage 1 components
│   ├── stage2_model_architecture/    # Stage 2 components
│   ├── stage3_training/              # Stage 3 components
│   ├── stage4_inference/             # Stage 4 components
│   └── pipeline_orchestrator.py      # Main coordinator
├── config/
│   └── pipeline_config.yaml          # Configuration
├── utils/
│   ├── logger.py                     # Logging utilities
│   └── visualization.py              # Debug visualization
├── preprocess_dataset.py             # Batch preprocessing
├── train_sonar_complete.py           # Complete pipeline
├── train_sonar.py                   # Training only
├── inference.py                      # Inference only
└── test_pipeline.py                  # Stage testing
```

## Usage Workflow

### Complete Pipeline Execution
```bash
python train_sonar_complete.py --data sonar_detect-1/data.yaml --preprocess --eval
```

### Step-by-Step Execution
1. **Test Stage 1**: `python test_pipeline.py --image [path] --stage 1`
2. **Preprocess Dataset**: `python preprocess_dataset.py --data [data.yaml]`
3. **Train Model**: `python train_sonar.py --data [preprocessed_data.yaml]`
4. **Evaluate**: `python evaluate_model.py --model [model.pt] --data [data.yaml]`
5. **Inference**: `python inference.py --image [path] --model [model.pt] --visualize`

## Technical Specifications

### Dependencies
- **PyWavelets**: Wavelet denoising
- **scikit-image**: CLAHE normalization
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation
- **OpenCV**: Image I/O and processing
- **NumPy**: Numerical operations

### Performance Considerations
- **Preprocessing**: Done once per dataset (saved to disk)
- **Batch Processing**: Processes entire dataset in single run
- **Memory Management**: Processes images one at a time to avoid memory issues
- **Progress Tracking**: Uses tqdm for visual progress bars

## Validation and Testing

### Testing Strategy
1. **Unit Testing**: Individual components tested with `test_pipeline.py`
2. **Integration Testing**: Full pipeline tested with `train_sonar_complete.py`
3. **Visual Validation**: Visualization tools for each stage output
4. **Error Tracking**: Comprehensive logging for debugging

### Quality Assurance
- Input validation at each stage
- Error handling with informative messages
- Dimension checking for array operations
- Path validation for file operations

## Results and Outputs

### Stage 1 Outputs
- Preprocessed images (denoised, normalized)
- Bottom line coordinates
- Blind zone masks
- Altitude measurements

### Stage 2 Outputs
- Trained model weights
- Training curves and metrics
- Validation results

### Stage 4 Outputs
- Detection coordinates (pixel space)
- Geocoded coordinates (real-world space)
- Confidence scores
- Class predictions

## Future Work

### Full Integration Required
1. **Stage 2**: Modify Ultralytics source to integrate SPD-Conv, EMA, BiFPN
2. **Stage 3**: Custom training loop for Shape-IoU loss and Nadir masking
3. **P2 Detection Head**: Enable high-resolution detection grid

### Optimization Opportunities
1. Parallel preprocessing (multiprocessing)
2. GPU acceleration for preprocessing
3. Caching of intermediate results
4. Incremental preprocessing (resume from checkpoint)

## Conclusion

The pipeline implementation successfully integrates physics-based signal processing with deep learning, following the system design specifications. Each stage is modular, testable, and can be configured independently. The implementation provides a solid foundation for sonar object detection with proper error tracking and debugging capabilities.


