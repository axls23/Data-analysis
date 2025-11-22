"""
Model Architectures for Facial Emotion Recognition
====================================================

This package contains multiple CNN architectures for emotion classification:
- MobileNetV2: Lightweight, efficient for mobile/edge deployment
- ResNet18/50: Strong baseline with residual connections
- EfficientNet-B0: State-of-the-art efficiency

All models use transfer learning with ImageNet pretrained weights and
custom classification heads for 7-emotion classification.

Usage:
    from models import create_model
    
    model = create_model('mobilenet', num_classes=7, pretrained=True)
    model = create_model('resnet18', num_classes=7, pretrained=True)
    model = create_model('efficientnet', num_classes=7, pretrained=True)
"""

from .base_model import BaseEmotionModel
from .mobilenet import MobileNetV2Emotion
from .resnet import ResNet18Emotion, ResNet50Emotion
from .efficientnet import EfficientNetB0Emotion
from .model_factory import create_model, list_available_models

__all__ = [
    'BaseEmotionModel',
    'MobileNetV2Emotion',
    'ResNet18Emotion',
    'ResNet50Emotion',
    'EfficientNetB0Emotion',
    'create_model',
    'list_available_models',
]

__version__ = '1.0.0'
