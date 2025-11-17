@echo off
echo ======================================================================
echo EXPRESSION DETECTION - INFERENCE ON SNAPSHOTS
echo ======================================================================
echo.
echo Step 1: Make sure you have snapshots in the 'snapshots' folder
echo         (from video mode or manually placed images)
echo.
echo Step 2: Running batch inference on snapshots...
echo.

python expression-detection-optimized.py --mode inference --model_path models/best_model.pth --input_folder snapshots --output_folder inference_results

echo.
echo ======================================================================
echo Inference complete! Check 'inference_results' folder for results.
echo ======================================================================
pause




