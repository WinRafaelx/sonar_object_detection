# Complete Pipeline Guide: Preprocess → Train → Evaluate

This guide shows you how to run the **complete pipeline** with preprocessing, training, and evaluation.

## Overview

The complete pipeline consists of three main steps:

1. **Preprocessing (Stage 1)**: Apply signal conditioning to all images
2. **Training (Stage 2)**: Train YOLO-Sonar model on preprocessed images
3. **Evaluation**: Evaluate model performance on test set

## Option 1: All-in-One Script (Recommended)

Run everything in one command:

```bash
python train_sonar_complete.py --data sonar_detect-1/data.yaml --preprocess --eval
```

**What this does:**
1. Loads dataset from `sonar_detect-1/data.yaml`
2. Preprocesses all images (Stage 1: Viterbi, Wavelet, Normalization)
3. Trains model on preprocessed images
4. Evaluates model on test set

**Output:**
- Preprocessed dataset: `runs/train/yolo_sonar/preprocessed_dataset/`
- Trained model: `runs/train/yolo_sonar/weights/best.pt`
- Evaluation results: Console output + plots

---

## Option 2: Step-by-Step (For Debugging)

### Step 1: Preprocess Dataset

Preprocess all training/validation/test images:

```bash
python preprocess_dataset.py --data sonar_detect-1/data.yaml --output dataset_preprocessed
```

**What this does:**
- Applies Stage 1 preprocessing to all images
- Saves preprocessed images to `dataset_preprocessed/`
- Copies labels (unchanged)
- Creates new `data.yaml` pointing to preprocessed images

**Output structure:**
```
dataset_preprocessed/
├── data.yaml
├── train/
│   ├── images/  (preprocessed)
│   └── labels/  (copied)
├── val/
│   ├── images/  (preprocessed)
│   └── labels/  (copied)
└── test/
    ├── images/  (preprocessed)
    └── labels/  (copied)
```

### Step 2: Train Model

Train on preprocessed images:

```bash
python train_sonar.py --data dataset_preprocessed/data.yaml --epochs 50
```

**What this does:**
- Loads preprocessed dataset
- Trains YOLO-Sonar model
- Saves best model to `runs/train/yolo_sonar/weights/best.pt`

### Step 3: Evaluate Model

Evaluate trained model:

```bash
python evaluate_model.py --model runs/train/yolo_sonar/weights/best.pt --data dataset_preprocessed/data.yaml --split test
```

**What this does:**
- Loads trained model
- Runs evaluation on test set
- Prints metrics (mAP50, mAP50-95, Precision, Recall)
- Saves evaluation plots

---

## Complete Example Workflow

Here's a complete example from start to finish:

```bash
# 1. Test Stage 1 on a single image (verify preprocessing works)
python test_pipeline.py --image sonar_detect-1/train/images/000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg --stage 1

# 2. Preprocess entire dataset
python preprocess_dataset.py --data sonar_detect-1/data.yaml --output dataset_preprocessed

# 3. Train model on preprocessed images
python train_sonar.py --data dataset_preprocessed/data.yaml --epochs 50 --batch 16

# 4. Evaluate model
python evaluate_model.py --model runs/train/yolo_sonar/weights/best.pt --data dataset_preprocessed/data.yaml --split test

# 5. Run inference on new images
python inference.py --image [test_image].jpg --model runs/train/yolo_sonar/weights/best.pt --visualize
```

---

## Understanding the Pipeline

### Stage 1: Preprocessing (Signal Conditioning)

**Applied to each image:**
1. **Viterbi Bottom Tracking**: Detects seabed line
2. **Wavelet Denoising**: Removes speckle noise
3. **Normalization**: Log transform + CLAHE

**Why preprocess before training?**
- Model learns on clean, standardized images
- Consistent preprocessing across train/val/test
- Better feature extraction for sonar-specific challenges

### Stage 2: Training

**What happens:**
- Model trains on preprocessed images
- Uses YOLO-Sonar architecture (with SPD-Conv, EMA, BiFPN if integrated)
- Saves checkpoints and best model

### Stage 3: Evaluation

**Metrics computed:**
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Per-class mAP**: Performance for each class

---

## Command Reference

### Preprocess Dataset
```bash
python preprocess_dataset.py \
    --data sonar_detect-1/data.yaml \
    --output dataset_preprocessed \
    --config config/pipeline_config.yaml
```

### Train Model
```bash
python train_sonar.py \
    --data dataset_preprocessed/data.yaml \
    --epochs 50 \
    --batch 16 \
    --model yolo11m.pt
```

### Evaluate Model
```bash
python evaluate_model.py \
    --model runs/train/yolo_sonar/weights/best.pt \
    --data dataset_preprocessed/data.yaml \
    --split test \
    --conf 0.25 \
    --iou 0.45
```

### Complete Pipeline (All-in-One)
```bash
python train_sonar_complete.py \
    --data sonar_detect-1/data.yaml \
    --preprocess \
    --eval \
    --epochs 50 \
    --batch 16
```

---

## Troubleshooting

### Problem: "Preprocessing takes too long"
**Solution:** Preprocessing is done once. After preprocessing, you can train multiple times without re-preprocessing.

### Problem: "Out of memory during preprocessing"
**Solution:** Process images in batches or reduce image size in config.

### Problem: "Model performs poorly"
**Solution:** 
- Check if preprocessing is working correctly (use `test_pipeline.py`)
- Verify preprocessed images look correct
- Try different hyperparameters (epochs, batch size, learning rate)

### Problem: "Evaluation metrics are low"
**Solution:**
- Train for more epochs
- Check if dataset is balanced
- Verify labels are correct
- Try different confidence/IoU thresholds

---

## File Structure After Running

```
sonar_object_detection/
├── dataset_preprocessed/          # Preprocessed dataset
│   ├── data.yaml
│   ├── train/images/
│   ├── val/images/
│   └── test/images/
├── runs/train/yolo_sonar/        # Training outputs
│   ├── weights/
│   │   ├── best.pt               # Best model
│   │   └── last.pt                # Last checkpoint
│   ├── results.png                # Training curves
│   └── confusion_matrix.png
└── logs/                          # Detailed logs
    └── pipeline_*.log
```

---

## Next Steps

1. **Tune Hyperparameters**: Adjust epochs, batch size, learning rate
2. **Experiment with Preprocessing**: Modify `config/pipeline_config.yaml`
3. **Test on Different Images**: Use `inference.py` on new sonar images
4. **Analyze Results**: Review evaluation metrics and confusion matrix

---

## Quick Reference

| Task | Command |
|------|---------|
| Test preprocessing | `python test_pipeline.py --image [path] --stage 1` |
| Preprocess dataset | `python preprocess_dataset.py --data [data.yaml]` |
| Train model | `python train_sonar.py --data [preprocessed_data.yaml]` |
| Evaluate model | `python evaluate_model.py --model [model.pt] --data [data.yaml]` |
| Complete pipeline | `python train_sonar_complete.py --data [data.yaml] --preprocess --eval` |
| Run inference | `python inference.py --image [path] --model [model.pt] --visualize` |

