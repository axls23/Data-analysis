"""
MobileNetV2 for Emotion Recognition
=====================================

Lightweight architecture optimized for mobile and edge deployment.
Uses pretrained ImageNet weights with custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseEmotionModel


class MobileNetV2Emotion(BaseEmotionModel):
    """MobileNetV2 with custom emotion classification head"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True, 
                 hidden_dim: int = 128, dropout: float = 0.5):
        """
        Initialize MobileNetV2 for emotion recognition
        
        Args:
            num_classes: Number of emotion classes (default: 7)
            pretrained: Use ImageNet pretrained weights (default: True)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout: Dropout rate (default: 0.5)
        """
        super().__init__(num_classes, pretrained)
        
        # Load pretrained MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Get number of features from backbone
        # MobileNetV2 has 1280 features before classifier
        backbone_out_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head as per Phase 2 requirements:
        # Dense(128, ReLU) -> Dropout(0.5) -> Dense(num_classes)
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
        for idx, m in enumerate(self.classifier.modules()):
            if isinstance(m, nn.Linear):
                # Use smaller initialization for final output layer to prevent extreme logits
                if idx == len(list(self.classifier.modules())) - 1:
                    # Final layer: use smaller std for stable initial loss
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    # Hidden layers: use kaiming for ReLU
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
            num_layers: Number of layers to unfreeze from the end (-1 = all)
        """
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[INFO] Unfrozen all MobileNetV2 backbone layers")
        else:
            # MobileNetV2 has 'features' module with sequential blocks
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
            
            print(f"[INFO] Unfrozen last {num_layers} blocks of MobileNetV2 (blocks {start_idx}-{total_blocks-1})")
