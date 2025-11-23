"""
Base Model Class for Emotion Recognition
==========================================

Abstract base class that defines the common interface for all emotion recognition models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple


class BaseEmotionModel(ABC, nn.Module):
    """Abstract base class for emotion recognition models"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        """
        Initialize base model
        
        Args:
            num_classes: Number of emotion classes (default: 7)
            pretrained: Use pretrained weights (default: True)
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = None
        self.classifier = None
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    def freeze_backbone(self):
        """Freeze all backbone layers (for initial training)"""
        if self.backbone is None:
            return
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Set BatchNorm layers to eval mode to prevent updating running stats
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        print(f"[INFO] Frozen backbone layers (including BatchNorm eval mode)")
    
    def unfreeze_backbone(self, num_layers: int = -1):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (-1 = all)
        """
        if self.backbone is None:
            return
        
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[INFO] Unfrozen all backbone layers")
        else:
            # Unfreeze last N layers
            # This is model-specific and should be overridden in child classes
            print(f"[WARNING] Partial unfreezing not implemented for this model")
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self) -> Tuple[int, int]:
        """
        Get count of trainable and total parameters
        
        Returns:
            (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def print_summary(self):
        """Print model summary"""
        trainable, total = self.get_trainable_params()
        print(f"\n{'='*70}")
        print(f"{self.__class__.__name__} Summary")
        print(f"{'='*70}")
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters:    {total - trainable:,}")
        print(f"Trainable ratio:      {trainable/total*100:.2f}%")
        print(f"{'='*70}\n")
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.__class__.__name__
