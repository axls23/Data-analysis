"""Custom loss functions for emotion recognition training.

Includes:
- LabelSmoothingCrossEntropy: CrossEntropyLoss with label smoothing for regularization
- Prevents overconfident predictions and improves generalization on small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing regularization.
    
    Label smoothing prevents the model from becoming overconfident by
    smoothing the target distribution. Instead of hard targets (0 or 1),
    targets become (epsilon/(K-1)) for wrong classes and (1-epsilon) for
    correct class, where K is the number of classes.
    
    Research shows this improves generalization, especially on small datasets.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
                  0.0 = no smoothing (standard cross-entropy)
                  0.1 = recommended for most tasks
                  Higher values = more smoothing (use with caution)
        
    Example:
        >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> logits = model(inputs)
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.
        
        Args:
            pred: Model predictions (logits), shape (batch_size, num_classes)
            target: Ground truth class indices, shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Get log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Get number of classes
        num_classes = pred.size(-1)
        
        # Create smooth target distribution
        # Correct class gets (1 - smoothing), others get smoothing/(num_classes-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Compute KL divergence between smooth distribution and model predictions
        # Equivalent to cross-entropy but with smoothed targets
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance (optional alternative to label smoothing).
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    Useful when dataset has severe class imbalance.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor (default: 1.0, or tensor of class weights)
        gamma: Focusing parameter (default: 2.0)
               0 = standard cross-entropy
               Higher values = more focus on hard examples
        reduction: 'mean' or 'sum'
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (logits), shape (batch_size, num_classes)
            targets: Ground truth class indices, shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Gather probabilities for target classes
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - target_probs) ** self.gamma
        loss = -self.alpha * focal_weight * target_log_probs
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_type: str = 'cross_entropy', **kwargs):
    """Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'label_smoothing', 'focal')
        **kwargs: Additional arguments passed to loss constructor
                 - smoothing: for label_smoothing (default 0.1)
                 - alpha, gamma: for focal loss
    
    Returns:
        Loss function instance
    
    Example:
        >>> criterion = get_loss_function('label_smoothing', smoothing=0.1)
        >>> criterion = get_loss_function('focal', alpha=1.0, gamma=2.0)
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: cross_entropy, label_smoothing, focal")


__all__ = [
    'LabelSmoothingCrossEntropy',
    'FocalLoss',
    'get_loss_function'
]
