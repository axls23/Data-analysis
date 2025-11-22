"""
EfficientNet for Emotion Recognition
======================================

State-of-the-art efficient architecture with compound scaling.
Uses pretrained ImageNet weights with custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseEmotionModel


class EfficientNetB0Emotion(BaseEmotionModel):
    """EfficientNet-B0 with custom emotion classification head"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 hidden_dim: int = 128, dropout: float = 0.5):
        """
        Initialize EfficientNet-B0 for emotion recognition
        
        Args:
            num_classes: Number of emotion classes (default: 7)
            pretrained: Use ImageNet pretrained weights (default: True)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout: Dropout rate (default: 0.5)
        """
        super().__init__(num_classes, pretrained)
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get number of features from backbone
        # EfficientNet-B0 has 1280 features before classifier
        backbone_out_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        # Note: Backbone already outputs features after global pooling
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier layer weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def unfreeze_backbone(self, num_layers: int = -1):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            num_layers: Number of blocks to unfreeze from the end (-1 = all)
        """
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[INFO] Unfrozen all EfficientNet-B0 backbone layers")
        else:
            # EfficientNet has 'features' module with sequential blocks
            # Unfreeze last N blocks
            total_blocks = len(self.backbone.features)
            start_idx = max(0, total_blocks - num_layers)
            
            # First freeze all
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Then unfreeze last N blocks
            for i in range(start_idx, total_blocks):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = True
            
            print(f"[INFO] Unfrozen last {num_layers} blocks of EfficientNet-B0 (blocks {start_idx}-{total_blocks-1})")
