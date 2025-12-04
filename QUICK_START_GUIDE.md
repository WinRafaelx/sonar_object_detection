# Quick Start Guide: Running the Pipeline

## Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Roboflow API key (if downloading dataset):**
```bash
# Windows
set ROBOFLOW_API_KEY=your_api_key_here

# Linux/Mac
export ROBOFLOW_API_KEY=your_api_key_here
```

## Step-by-Step Workflow

### Step 1: Test Stage 1 (Preprocessing) - **START HERE**

Test the signal conditioning stage on a single image:

```bash
python test_pipeline.py --image sonar_detect-1/train/images/000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg --stage 1
```

**What this does:**
- Loads a sonar image
- Runs Viterbi bottom tracking
- Applies wavelet denoising
- Applies normalization
- Saves visualization to `test_output/stage1_test.png`

**Expected output:**
- Console logs showing each processing step
- Visualization image showing original vs processed
- Log file in `logs/` directory

**If successful:** You'll see a side-by-side comparison image showing the preprocessing results.

---

### Step 2: Train the Model

#### Option A: Use Existing Dataset (Recommended)
```bash
python train_sonar.py --data sonar_detect-1/data.yaml --epochs 50 --batch 16
```

#### Option B: Download Dataset First
```bash
python train_sonar.py --download-roboflow --epochs 50 --batch 16
```

**What this does:**
- Loads or downloads the dataset
- Initializes YOLO-Sonar model
- Trains the model with your dataset
- Saves best model to `runs/train/yolo_sonar/weights/best.pt`

**Expected output:**
- Training progress bars
- Loss curves
- Model checkpoints saved periodically
- Best model saved at the end

**Training time:** Depends on your GPU/CPU, typically 1-4 hours for 50 epochs

---

### Step 3: Run Inference (Complete Pipeline)

After training, test the complete pipeline on an image:

```bash
python inference.py --image sonar_detect-1/test/images/[any_test_image].jpg --model runs/train/yolo_sonar/weights/best.pt --visualize
```

**What this does:**
1. **Stage 1:** Preprocesses the image (bottom tracking, denoising, normalization)
2. **Stage 2:** Runs model inference (with SAHI slicing if enabled)
3. **Stage 4:** Merges detections (WBF) and geocodes to real-world coordinates
4. Saves results to `output/` directory

**Expected output:**
- `output/detections.txt` - Detection coordinates
- `output/geocoded_detections.txt` - Real-world coordinates (if geocoding enabled)
- `output/stage1_processing.png` - Preprocessing visualization
- `output/detections.png` - Detection visualization

---

## Complete Example Workflow

Here's a complete example from start to finish:

```bash
# 1. Test preprocessing on a sample image
python test_pipeline.py --image sonar_detect-1/train/images/000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg --stage 1

# 2. Train the model (if not already trained)
python train_sonar.py --data sonar_detect-1/data.yaml --epochs 50

# 3. Run inference on a test image
python inference.py --image sonar_detect-1/test/images/[test_image].jpg --model runs/train/yolo_sonar/weights/best.pt --visualize
```

---

## Troubleshooting

### Problem: "Could not load image"
**Solution:** Check the image path. Use absolute paths or paths relative to the project root.

### Problem: "Model not found"
**Solution:** Make sure you've trained the model first, or use the base model:
```bash
python inference.py --image [image] --model yolo11m.pt --visualize
```

### Problem: "ROBOFLOW_API_KEY not set"
**Solution:** Set the environment variable (see Prerequisites above)

### Problem: "No module named 'pipeline'"
**Solution:** Make sure you're running from the project root directory:
```bash
cd C:\Users\meow7\Raf_dev\sonar_object_detection
python test_pipeline.py --image [path]
```

### Problem: Import errors
**Solution:** Install missing dependencies:
```bash
pip install PyWavelets scikit-image pyyaml matplotlib
```

---

## Understanding the Output

### Stage 1 Test Output
- **Console:** Shows processing steps, bottom line detection, altitude measurements
- **Visualization:** Side-by-side original vs processed image with seabed line overlay
- **Log file:** Detailed debug information in `logs/pipeline_*.log`

### Training Output
- **Console:** Training progress, loss values, validation metrics
- **Files:** Model weights saved to `runs/train/yolo_sonar/weights/`
- **Plots:** Training curves saved to `runs/train/yolo_sonar/`

### Inference Output
- **detections.txt:** Format: `x1 y1 x2 y2 confidence class`
- **geocoded_detections.txt:** Format: `x1 y1 x2 y2 confidence class ground_x ground_y`
- **Visualizations:** Images showing detections overlaid on processed image

---

## Next Steps

1. **Tune Parameters:** Edit `config/pipeline_config.yaml` to adjust preprocessing and model settings
2. **Test Different Images:** Try various sonar images to see how the pipeline performs
3. **Analyze Results:** Check the log files in `logs/` for detailed processing information
4. **Customize:** Modify individual stage components as needed for your specific use case

---

## Quick Reference

| Task | Command |
|------|---------|
| Test Stage 1 | `python test_pipeline.py --image [path] --stage 1` |
| Train Model | `python train_sonar.py --data sonar_detect-1/data.yaml` |
| Run Inference | `python inference.py --image [path] --model [model.pt] --visualize` |
| Download Dataset | `python train_sonar.py --download-roboflow` |

---

## Need Help?

- Check `README_PIPELINE.md` for detailed documentation
- Check `DATASET_INFO.md` for dataset information
- Check log files in `logs/` directory for error details
- Review `config/pipeline_config.yaml` for configuration options

