## Plan: Modular Restructuring of Expression Detection Pipeline

This plan divides the 1447-line monolithic expression-detection-optimized.py into 6 clean modules, each assigned to one team member. The structure ensures minimal dependencies and clear separation of concerns.

### Steps:

1. **Create config.py (Team Member 1)**
    
    Extract check_gpu_availability, Config class, and global variables (YOLO_AVAILABLE, CUDA_AVAILABLE). No dependencies on other project modules.

2. **Create data.py (Team Member 2)**
    
    Extract ExpressionDataset class and transform objects (train_transform, val_transform, inference_transform). Import from config.py only.

3. **Create model.py (Team Member 3)**
    
    Extract create_model function supporting EfficientNet-B0, ResNet50, and MobileNetV3 architectures. Import from config.py only.

4. **Create train.py (Team Member 4)**
    
    Extract train_epoch, validate_epoch, tune_hyperparams, and train_model functions. Import from config, data, model, and metrics.

5. **Create metrics.py (Team Member 5)**
    
    Extract compute_metrics, plot_confusion_matrix, plot_training_curves, and print_metrics_report. Import from config.py only.

6. **Create inference.py and main.py (Team Member 6)**
    
    Extract ModelEvaluator and InferenceEngine classes into inference.py. Create main.py with the main CLI function orchestrating all workflows (train/evaluate/inference modes).
