@echo off
REM Example batch script for Windows to run the pipeline
REM Modify paths as needed

echo ========================================
echo Sonar Object Detection Pipeline
echo ========================================
echo.

REM Step 1: Test Stage 1 preprocessing
echo [Step 1] Testing Stage 1: Signal Conditioning...
python test_pipeline.py --image sonar_detect-1\train\images\000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg --stage 1
if errorlevel 1 (
    echo ERROR: Stage 1 test failed!
    pause
    exit /b 1
)
echo Stage 1 test completed successfully!
echo.

REM Step 2: Train model (uncomment to run)
REM echo [Step 2] Training model...
REM python train_sonar.py --data sonar_detect-1\data.yaml --epochs 50 --batch 16
REM if errorlevel 1 (
REM     echo ERROR: Training failed!
REM     pause
REM     exit /b 1
REM )
REM echo Training completed!
REM echo.

REM Step 3: Run inference (uncomment after training)
REM echo [Step 3] Running inference...
REM python inference.py --image sonar_detect-1\test\images\[test_image].jpg --model runs\train\yolo_sonar\weights\best.pt --visualize
REM if errorlevel 1 (
REM     echo ERROR: Inference failed!
REM     pause
REM     exit /b 1
REM )
REM echo Inference completed!
REM echo.

echo ========================================
echo Pipeline execution complete!
echo ========================================
pause

