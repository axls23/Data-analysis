"""
Model Configuration
====================

Default hyperparameters for emotion recognition models.
"""

# Dataset configuration
NUM_CLASSES = 7
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
INPUT_SIZE = 224  # Input image size (224x224)

# Model architecture configuration
HIDDEN_DIM = 128  # Hidden layer dimension in classifier
DROPOUT_RATE = 0.5  # Dropout rate in classifier

# Training configuration (to be used in Phase 3)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Initial learning rate for warm-up (reduced from 1e-3 to fix underfitting)
WEIGHT_DECAY = 1e-4  # L2 regularization (research-backed for small datasets)
FINE_TUNE_LR = 3e-5  # Learning rate for fine-tuning (increased from 1e-5)
NUM_EPOCHS_WARMUP = 20  # Epochs for initial training with frozen backbone
NUM_EPOCHS_FINETUNE = 20  # Epochs for fine-tuning with unfrozen layers
EARLY_STOPPING_PATIENCE = 20  # Stop if no improvement for N epochs (warmup phase)
FINETUNE_PATIENCE = 5  # Early stopping patience for fine-tuning (tighter convergence)

# Data augmentation configuration (to be used in Phase 3)
# TWO-STAGE AUGMENTATION STRATEGY:
# - WARMUP (frozen backbone): Moderate augmentation - model has limited capacity
# - FINE-TUNING (unfrozen): Aggressive augmentation - model can handle complexity

# WARMUP AUGMENTATION (Phase 3a) - Conservative for frozen backbone
AUGMENTATION_WARMUP = {
    'rotation_range': 10,  # Moderate rotation
    'width_shift_range': 0.1,  # Moderate shift
    'height_shift_range': 0.1,  # Moderate shift
    'horizontal_flip': True,  # Always beneficial
    'brightness_range': (0.8, 1.2),  # Conservative brightness
    'contrast_range': 0.2,  # Moderate contrast
    'saturation_range': 0.2,  # Moderate saturation
    'hue_range': 0.05,  # Slight hue variation
    'zoom_range': 0.1,  # Moderate zoom
    'resized_crop_scale': (0.9, 1.0),  # Conservative crop
    'random_erasing_prob': 0.2,  # Light occlusion
    'random_erasing_scale': (0.02, 0.1),  # Small patches
    'perspective_distortion': 0.0,  # DISABLED for warmup
    'perspective_prob': 0.0,  # DISABLED for warmup
    'gaussian_blur_prob': 0.0,  # DISABLED for warmup
    'grayscale_prob': 0.0,  # DISABLED for warmup
}

# FINE-TUNING AUGMENTATION (Phase 3b) - Adjusted for small dataset
# Reduced aggression to fix underfitting in ResNet50/18
AUGMENTATION_FINETUNE = {
    'rotation_range': 12,  # Reduced from 15
    'width_shift_range': 0.12,  # Reduced from 0.15
    'height_shift_range': 0.12,  # Reduced from 0.15
    'horizontal_flip': True,
    'brightness_range': (0.75, 1.25),  # Slightly less aggressive
    'contrast_range': 0.25,  # Slightly less aggressive
    'saturation_range': 0.25,  # Slightly less aggressive
    'hue_range': 0.08,  # Reduced from 0.1
    'zoom_range': 0.12,  # Reduced from 0.15
    'resized_crop_scale': (0.88, 1.12),  # Tighter crop range (was 0.85-1.15)
    'random_erasing_prob': 0.3,  # Reduced from 0.4
    'random_erasing_scale': (0.02, 0.12),  # Smaller patches
    'perspective_distortion': 0.15,  # Reduced from 0.2
    'perspective_prob': 0.2,  # Reduced from 0.3
    'gaussian_blur_prob': 0.1,  # Reduced from 0.2
    'grayscale_prob': 0.1,  # Kept same
}

# Default augmentation (backward compatibility - use warmup settings)
AUGMENTATION = AUGMENTATION_WARMUP

# Model-specific hyperparameters for optimal performance
# Different models need different regularization based on capacity
MODEL_SPECIFIC_PARAMS = {
    'mobilenet': {
        'dropout': 0.4,  # Lighter dropout for smaller model
        'weight_decay': 1e-4,  # Standard L2 regularization
        'fine_tune_lr_multiplier': 1.0,  # Use base FINE_TUNE_LR
        'unfreeze_blocks_stage2': 4,  # Unfreeze last 4 inverted residual blocks
    },
    'efficientnet': {
        'dropout': 0.5,  # Moderate dropout
        'weight_decay': 1e-4,  # Standard L2 regularization
        'fine_tune_lr_multiplier': 1.0,  # Use base FINE_TUNE_LR
        'unfreeze_blocks_stage2': 5,  # Unfreeze last 5 MBConv blocks
    },
    'resnet18': {
        'dropout': 0.5,  # Moderate dropout
        'weight_decay': 5e-5,  # Slightly less regularization (residual connections help)
        'fine_tune_lr_multiplier': 2.0,  # Can tolerate higher LR due to residual connections
        'unfreeze_layers_stage2': [3, 4],  # Unfreeze layer3 + layer4 (half the network)
    },
    'resnet50': {
        'dropout': 0.6,  # Heavier dropout for larger model
        'weight_decay': 1e-5,  # Less weight decay (larger model needs less regularization)
        'fine_tune_lr_multiplier': 1.0,  # Use base FINE_TUNE_LR
        'unfreeze_layers_stage2': [4],  # Unfreeze only layer4 initially (conservative)
        'unfreeze_blocks_layer3': 2,  # Then unfreeze last 2 blocks of layer3
    },
}

# Label smoothing for regularization (prevents overconfident predictions)
LABEL_SMOOTHING = 0.1  # Smooth labels with 0.1 epsilon

# Class weights for handling performance imbalance
# Based on observed F1 scores: neutral (47-70%), fear (55-62%), sad (63-75%)
# Order: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.5, 1.2, 1.0]
# neutral: 1.5x (worst performer, critical for improvement)
# sad: 1.2x (moderate underperformer)
# others: 1.0x (adequate performance)

# Cosine annealing learning rate schedule for fine-tuning
COSINE_T_MAX = 15  # Cosine annealing period (epochs)
COSINE_ETA_MIN = 1e-6  # Minimum learning rate for cosine annealing (increased from 1e-7)

# Paths (relative to project root)
PREPROCESSED_DATA_DIR = 'preprocessed_data'
DATA_SPLITS_DIR = 'data_splits'
RESULTS_DIR = 'results'
MODELS_DIR = 'saved_models'

# Device configuration
USE_GPU = True  # Use GPU if available
GPU_ID = 0  # GPU device ID (0 for NVIDIA primary GPU)
