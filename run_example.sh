#!/bin/bash
# Example shell script for Linux/Mac to run the pipeline
# Modify paths as needed

echo "========================================"
echo "Sonar Object Detection Pipeline"
echo "========================================"
echo ""

# Step 1: Test Stage 1 preprocessing
echo "[Step 1] Testing Stage 1: Signal Conditioning..."
python test_pipeline.py --image sonar_detect-1/train/images/000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg --stage 1
if [ $? -ne 0 ]; then
    echo "ERROR: Stage 1 test failed!"
    exit 1
fi
echo "Stage 1 test completed successfully!"
echo ""

# Step 2: Train model (uncomment to run)
# echo "[Step 2] Training model..."
# python train_sonar.py --data sonar_detect-1/data.yaml --epochs 50 --batch 16
# if [ $? -ne 0 ]; then
#     echo "ERROR: Training failed!"
#     exit 1
# fi
# echo "Training completed!"
# echo ""

# Step 3: Run inference (uncomment after training)
# echo "[Step 3] Running inference..."
# python inference.py --image sonar_detect-1/test/images/[test_image].jpg --model runs/train/yolo_sonar/weights/best.pt --visualize
# if [ $? -ne 0 ]; then
#     echo "ERROR: Inference failed!"
#     exit 1
# fi
# echo "Inference completed!"
# echo ""

echo "========================================"
echo "Pipeline execution complete!"
echo "========================================"

