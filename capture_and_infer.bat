@echo off
echo ======================================================================
echo EXPRESSION DETECTION - CAPTURE SNAPSHOTS AND RUN INFERENCE
echo ======================================================================
echo.
echo This script will:
echo   1. Capture snapshots from webcam (saves to 'snapshots' folder)
echo   2. Run batch inference on captured snapshots
echo.
echo Press any key to start video capture (press 'q' in video window to stop)...
pause

echo.
echo Starting video capture...
python expression-detection-optimized.py --mode inference --model_path models/best_model.pth --video_source 0 --save_dir snapshots --frame_interval 3

echo.
echo Video capture complete. Running inference on snapshots...
echo.

python expression-detection-optimized.py --mode inference --model_path models/best_model.pth --input_folder snapshots --output_folder inference_results

echo.
echo ======================================================================
echo Complete! Check 'inference_results' folder for results.
echo ======================================================================
pause





