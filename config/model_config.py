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
LEARNING_RATE = 1e-3  # Initial learning rate for warm-up
FINE_TUNE_LR = 1e-5  # Learning rate for fine-tuning
NUM_EPOCHS_WARMUP = 20  # Epochs for initial training with frozen backbone
NUM_EPOCHS_FINETUNE = 20  # Epochs for fine-tuning with unfrozen layers
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N epochs

# Data augmentation configuration (to be used in Phase 3)
AUGMENTATION = {
    'rotation_range': 10,  # Rotation in degrees
    'width_shift_range': 0.1,  # Horizontal shift as fraction
    'height_shift_range': 0.1,  # Vertical shift as fraction
    'horizontal_flip': True,  # Random horizontal flip
    'brightness_range': (0.8, 1.2),  # Brightness adjustment
    'zoom_range': 0.1,  # Random zoom
}

# Paths (relative to project root)
PREPROCESSED_DATA_DIR = 'preprocessed_data'
DATA_SPLITS_DIR = 'data_splits'
RESULTS_DIR = 'results'
MODELS_DIR = 'saved_models'

# Device configuration
USE_GPU = True  # Use GPU if available
GPU_ID = 0  # GPU device ID (0 for NVIDIA primary GPU)
