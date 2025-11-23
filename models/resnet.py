"""
ResNet Architectures for Emotion Recognition
==============================================

ResNet18 and ResNet50 with custom emotion classification heads.
Uses pretrained ImageNet weights with custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseEmotionModel


class ResNet18Emotion(BaseEmotionModel):
    """ResNet18 with custom emotion classification head"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 hidden_dim: int = 128, dropout: float = 0.5):
        """
        Initialize ResNet18 for emotion recognition
        
        Args:
            num_classes: Number of emotion classes (default: 7)
            pretrained: Use ImageNet pretrained weights (default: True)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout: Dropout rate (default: 0.5)
        """
        super().__init__(num_classes, pretrained)
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Get number of features from backbone
        # ResNet18 has 512 features before fc layer
        backbone_out_features = self.backbone.fc.in_features
        
        # Remove original fc layer
        self.backbone.fc = nn.Identity()
        
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
        """Initialize classifier layer weights - ResNet18"""
        linear_layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        for idx, m in enumerate(linear_layers):
            # Use smaller initialization for final output layer to prevent extreme logits
            if idx == len(linear_layers) - 1:
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
            num_layers: Number of layer groups to unfreeze from the end (-1 = all)
                       ResNet has 4 layer groups (layer1, layer2, layer3, layer4)
        """
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[INFO] Unfrozen all ResNet18 backbone layers")
        else:
            # ResNet has layer1, layer2, layer3, layer4
            layer_groups = [self.backbone.layer1, self.backbone.layer2, 
                           self.backbone.layer3, self.backbone.layer4]
            
            # First freeze all
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layer groups
            start_idx = max(0, len(layer_groups) - num_layers)
            for i in range(start_idx, len(layer_groups)):
                for param in layer_groups[i].parameters():
                    param.requires_grad = True
            
            print(f"[INFO] Unfrozen last {num_layers} layer group(s) of ResNet18 (layer{start_idx+1}-layer4)")


class ResNet50Emotion(BaseEmotionModel):
    """ResNet50 with custom emotion classification head"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 hidden_dim: int = 128, dropout: float = 0.5):
        """
        Initialize ResNet50 for emotion recognition
        
        Args:
            num_classes: Number of emotion classes (default: 7)
            pretrained: Use ImageNet pretrained weights (default: True)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout: Dropout rate (default: 0.5)
        """
        super().__init__(num_classes, pretrained)
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get number of features from backbone
        # ResNet50 has 2048 features before fc layer
        backbone_out_features = self.backbone.fc.in_features
        
        # Remove original fc layer
        self.backbone.fc = nn.Identity()
        
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
        """Initialize classifier layer weights - ResNet50"""
        linear_layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        for idx, m in enumerate(linear_layers):
            # Use smaller initialization for final output layer to prevent extreme logits
            if idx == len(linear_layers) - 1:
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
            num_layers: Number of layer groups to unfreeze from the end (-1 = all)
                       ResNet has 4 layer groups (layer1, layer2, layer3, layer4)
        """
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[INFO] Unfrozen all ResNet50 backbone layers")
        else:
            # ResNet has layer1, layer2, layer3, layer4
            layer_groups = [self.backbone.layer1, self.backbone.layer2,
                           self.backbone.layer3, self.backbone.layer4]
            
            # First freeze all
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layer groups
            start_idx = max(0, len(layer_groups) - num_layers)
            for i in range(start_idx, len(layer_groups)):
                for param in layer_groups[i].parameters():
                    param.requires_grad = True
            
            print(f"[INFO] Unfrozen last {num_layers} layer group(s) of ResNet50 (layer{start_idx+1}-layer4)")
