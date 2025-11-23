"""
Configuration Package
======================

Central configuration for the facial emotion recognition project.
"""

from .model_config import *

__all__ = [
    'NUM_CLASSES',
    'EMOTION_LABELS',
    'INPUT_SIZE',
    'HIDDEN_DIM',
    'DROPOUT_RATE',
    'BATCH_SIZE',
    'LEARNING_RATE',
    'FINE_TUNE_LR',
    'NUM_EPOCHS_WARMUP',
    'NUM_EPOCHS_FINETUNE',
    'EARLY_STOPPING_PATIENCE',
    'AUGMENTATION',
    'PREPROCESSED_DATA_DIR',
    'DATA_SPLITS_DIR',
    'RESULTS_DIR',
    'MODELS_DIR',
    'USE_GPU',
    'GPU_ID',
    'IMAGE_SIZE',
]
