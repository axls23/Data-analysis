"""
Model Factory
==============

Factory pattern for creating emotion recognition models.
"""

from typing import Optional
from .mobilenet import MobileNetV2Emotion
from .resnet import ResNet18Emotion, ResNet50Emotion
from .efficientnet import EfficientNetB0Emotion


# Model registry
MODEL_REGISTRY = {
    'mobilenet': MobileNetV2Emotion,
    'mobilenetv2': MobileNetV2Emotion,
    'resnet18': ResNet18Emotion,
    'resnet50': ResNet50Emotion,
    'efficientnet': EfficientNetB0Emotion,
    'efficientnet_b0': EfficientNetB0Emotion,
}


def create_model(model_name: str, num_classes: int = 7, pretrained: bool = True,
                 hidden_dim: int = 128, dropout: float = 0.5):
    """
    Create an emotion recognition model by name
    
    Args:
        model_name: Name of model architecture
                   Options: 'mobilenet', 'resnet18', 'resnet50', 'efficientnet'
        num_classes: Number of emotion classes (default: 7)
        pretrained: Use ImageNet pretrained weights (default: True)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout rate (default: 0.5)
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model_name is not recognized
    
    Examples:
        >>> model = create_model('mobilenet')
        >>> model = create_model('resnet18', num_classes=7, pretrained=True)
        >>> model = create_model('efficientnet', hidden_dim=256, dropout=0.3)
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(sorted(set(MODEL_REGISTRY.keys())))
        raise ValueError(f"Unknown model: '{model_name}'. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name_lower]
    model = model_class(
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    return model


def list_available_models():
    """
    List all available model architectures
    
    Returns:
        List of model names
    """
    # Return unique model names
    unique_models = sorted(set(MODEL_REGISTRY.keys()))
    return unique_models


def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific model
    
    Args:
        model_name: Name of model architecture
    
    Returns:
        Dictionary with model information
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    
    model_class = MODEL_REGISTRY[model_name_lower]
    
    # Create temporary model to get info
    temp_model = model_class(num_classes=7, pretrained=False)
    trainable, total = temp_model.get_trainable_params()
    
    info = {
        'name': model_class.__name__,
        'class': model_class,
        'total_params': total,
        'description': model_class.__doc__.strip() if model_class.__doc__ else '',
    }
    
    return info
