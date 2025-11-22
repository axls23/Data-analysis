#!/usr/bin/env python3
"""
Optimized Expression Detection Pipeline
========================================

Structured Workflow:
1. TRAIN: Train model with comprehensive evaluation metrics
2. EVALUATE: Test model on separate test dataset
3. COMPARE: Compare multiple trained models
4. INFERENCE: Real-time video processing with face detection

Key Features:
- F1 Score, Precision, Recall metrics (per-class, macro, weighted)
- Confusion matrix visualization
- Training curves plotting
- YOLOv8 integration for comparison
- Clear separation of training/testing/inference phases
- Best model selection based on F1 score

Author: AI Assistant
Date: November 2025
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import json
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# YOLO for face detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Check GPU availability with detailed diagnostics
def check_gpu_availability(verbose=True):
    """Comprehensive GPU availability check"""
    cuda_available = torch.cuda.is_available()
    
    # Only print if verbose (avoid printing in worker processes)
    if verbose:
        print("=" * 70)
        print("GPU DETECTION CHECK")
        print("=" * 70)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Test GPU with a simple tensor operation
            try:
                test_tensor = torch.randn(1, 1).cuda()
                print(f"GPU test: SUCCESS - Tensor created on {test_tensor.device}")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"GPU test: FAILED - {e}")
                cuda_available = False
        else:
            print("WARNING: CUDA not available. Training will use CPU (much slower).")
            print("To use GPU, ensure:")
            print("  1. NVIDIA GPU is installed")
            print("  2. CUDA drivers are installed (check with: nvidia-smi)")
            print("  3. PyTorch with CUDA support is installed")
        
        print("=" * 70)
    
    return cuda_available

# Run GPU check at import (only in main process)
# Check if we're in a worker process by checking for multiprocessing context
import multiprocessing
if __name__ == '__main__' or not hasattr(multiprocessing.current_process(), 'name') or multiprocessing.current_process().name == 'MainProcess':
    CUDA_AVAILABLE = check_gpu_availability(verbose=True)
else:
    CUDA_AVAILABLE = check_gpu_availability(verbose=False)


class Config:
    """Central configuration for all modes"""
    EXPRESSIONS = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
    EXPR_CODES = {
        'happy': 'HA', 'sad': 'SA', 'angry': 'AN', 
        'surprised': 'SU', 'neutral': 'NE', 'fear': 'FE', 'disgust': 'DI'
    }
    
    MODEL_ARCHITECTURES = ['efficientnet_b0', 'resnet50', 'mobilenet_v3']
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    TRAIN_SPLIT = 0.8
    
    MODEL_SAVE_DIR = 'models'
    RESULTS_DIR = 'results'
    DATA_DIR = 'dataset'

# ============================================================================
# DATA TRANSFORMS
# ============================================================================

# Enhanced transforms with data augmentation for regularization
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),  # Added light rotation for regularization
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Added for regularization
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),  # Added cutout regularization
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_model(model_name='efficientnet_b0', n_classes=7, pretrained=True):
    """Create model architecture with transfer learning and regularization"""
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        num_features = model.classifier[1].in_features
        # Enhanced regularization: Higher dropout rates to combat overfitting
        model.classifier = nn.Sequential(
    nn.Dropout(p=0.6, inplace=False),      # Increased from 0.5 to 0.6
    nn.Linear(num_features, 128),
    nn.ReLU(inplace=False),
    nn.BatchNorm1d(128),                   # Added Batch Normalization
    nn.Dropout(p=0.5, inplace=False),      # Increased from 0.3 to 0.5
    nn.Linear(128, n_classes)
)

    elif model_name == 'resnet50':
        model = models.resnet50(
            weights='IMAGENET1K_V2' if pretrained else None
        )
        num_features = model.fc.in_features
        # Add regularization to ResNet
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),
            nn.Linear(256, n_classes)
        )
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(
            weights='IMAGENET1K_V2' if pretrained else None
        )
        num_features = model.classifier[3].in_features
        # Add regularization to MobileNetV3
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 128),
            nn.Hardswish(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.4),
            nn.Linear(128, n_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# ============================================================================
# DATASET
# ============================================================================

class ExpressionDataset(Dataset):
    """Dataset loader for expression images - optimized for speed"""
    def __init__(self, data_dir, transform=None, expressions=None):
        self.data_dir = data_dir
        self.transform = transform
        self.expressions = expressions or Config.EXPRESSIONS
        self.images = []
        self.labels = []
        self._load_images()
    
    def _load_images(self):
        """Load images from folder structure: data_dir/expression/image.jpg"""
        image_extensions = ('.jpg', '.jpeg', '.png')
        for expr_idx, expr in enumerate(self.expressions):
            expr_dir = os.path.join(self.data_dir, expr)
            if os.path.exists(expr_dir):
                # Use listdir with sorted for consistency
                valid_files = sorted([
                    f for f in os.listdir(expr_dir)
                    if f.lower().endswith(image_extensions)
                ])
                self.images.extend([os.path.join(expr_dir, f) for f in valid_files])
                self.labels.extend([expr_idx] * len(valid_files))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        # Optimized image loading - use lazy loading if possible
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image is corrupted, return a black image
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Train for one epoch - highly optimized for GPU speed"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        # Move to GPU with non_blocking for async transfer
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Use gradient accumulation for efficiency (fuse zero_grad with backward)
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Use mixed precision if available and enabled
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Fast accuracy calculation on GPU (no need to collect all predictions)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    # Return minimal data (no need for all predictions during training)
    return epoch_loss, epoch_acc, None, None

def validate_epoch(model, dataloader, criterion, device, use_amp=False):
    """Validate model with optional mixed precision - optimized for GPU"""
    model.eval()
    running_loss = 0.0
    # Keep tensors on GPU for vectorized operations
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Move to GPU with non_blocking for async transfer
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Vectorized operations on GPU
            probs = torch.softmax(outputs, dim=1)  # Vectorized softmax on GPU
            preds = torch.argmax(outputs, dim=1)    # Vectorized argmax on GPU
            
            all_preds.append(preds)    # Keep on GPU
            all_labels.append(labels)  # Keep on GPU
            all_probs.append(probs)    # Keep on GPU
    
    # Concatenate all tensors on GPU (vectorized)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Calculate accuracy on GPU (vectorized)
    epoch_acc = 100 * (all_preds == all_labels).float().mean().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Only move to CPU at the end (numpy conversion)
    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    all_probs = all_probs.cpu().numpy()
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

# ============================================================================
# EVALUATION METRICS (INCLUDING F1 SCORE)
# ============================================================================

def compute_metrics(y_true, y_pred, class_names):
    """
    Compute comprehensive evaluation metrics including F1 score
    
    Returns metrics dict with:
    - Per-class: precision, recall, F1 score
    - Macro averages: treats all classes equally
    - Weighted averages: accounts for class imbalance
    - Confusion matrix
    """
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    
    # Macro averages (treats all classes equally)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted averages (accounts for class imbalance)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved: {save_path}")

def plot_training_curves(history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training curves saved: {save_path}")

def print_metrics_report(metrics):
    """Print comprehensive metrics report"""
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"\n  Macro Averages (all classes treated equally):")
    print(f"    Precision: {metrics['precision_macro']:.4f}")
    print(f"    Recall:    {metrics['recall_macro']:.4f}")
    print(f"    F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"\n  Weighted Averages (accounting for class imbalance):")
    print(f"    Precision: {metrics['precision_weighted']:.4f}")
    print(f"    Recall:    {metrics['recall_weighted']:.4f}")
    print(f"    F1-Score:  {metrics['f1_weighted']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*70)
    
    for i, class_name in enumerate(metrics['class_names']):
        print(f"{class_name:<15} "
              f"{metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<12.4f} "
              f"{metrics['support_per_class'][i]:<10}")
    
    print(f"{'='*70}\n")
def tune_hyperparams(model_fn, train_loader, val_loader, location_tag):
    """
    Try hyperparameter optimization using Optuna.
    Args:
        model_fn: model function accepting hyperparameters as input
        train_loader, val_loader: dataloaders for train/val
        location_tag: str, workflow tracking tag
    Returns:
        best_params: dict, the best hyperparameters found
    """
    import optuna

    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.6)
        model = model_fn(dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        val_acc = train_and_evaluate(model, train_loader, val_loader, optimizer)  # implement this
        return val_acc

    print(f"[{location_tag}] Beginning Optuna hyperparameter search...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print(f"[{location_tag}] Best params: {study.best_params}")
    return study.best_params

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(data_dir, model_name='efficientnet_b0', epochs=10, 
                batch_size=32, lr=0.001, train_split=0.8, save_name='best_model',
                resume_from_checkpoint=True):
    """
    Main training function with comprehensive evaluation and checkpoint management
    
    Args:
        data_dir: Dataset directory (expects folders per class)
        model_name: Architecture to use
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        train_split: Train/val split ratio
        save_name: Name for saved model
        resume_from_checkpoint: Resume training from existing checkpoint if available
    
    Returns:
        model: Trained model
        metrics: Final validation metrics
        history: Training history
    """
    # Setup
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Force GPU detection check
    cuda_available = torch.cuda.is_available()
    
    # Additional verification: try to create a tensor on GPU
    if cuda_available:
        try:
            test = torch.zeros(1).cuda()
            del test
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"WARNING: CUDA check passed but GPU tensor creation failed: {e}")
            cuda_available = False
    
    device = torch.device('cuda' if cuda_available else 'cpu')
    
    # Check for mixed precision support
    use_amp = cuda_available and hasattr(torch.cuda, 'amp')
    scaler = GradScaler() if use_amp else None
    
    # Warn if CPU is being used
    if not cuda_available:
        print("\n" + "!" * 70)
        print("WARNING: Training will use CPU instead of GPU!")
        print("This will be MUCH slower. To use GPU:")
        print("  1. Ensure NVIDIA GPU drivers are installed")
        print("  2. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("  3. Verify GPU: python -c 'import torch; print(torch.cuda.is_available())'")
        print("!" * 70 + "\n")
    
    print(f"\n{'='*70}")
    print("TRAINING MODE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    # Load dataset
    full_dataset = ExpressionDataset(data_dir, transform=None)
    total_samples = len(full_dataset)
    
    if total_samples == 0:
        raise ValueError(f"No images found in {data_dir}. Check dataset structure.")
    
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    print(f"\nDataset Split:")
    print(f"  Total samples: {total_samples}")
    print(f"  Training: {train_size} ({train_split*100:.1f}%)")
    print(f"  Validation: {val_size} ({(1-train_split)*100:.1f}%)")
    print(f"{'='*70}")
    
    # Split dataset
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_samples), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with transforms
    train_dataset = torch.utils.data.Subset(
        ExpressionDataset(data_dir, transform=train_transform), 
        train_indices.indices
    )
    val_dataset = torch.utils.data.Subset(
        ExpressionDataset(data_dir, transform=val_transform), 
        val_indices.indices
    )
    
    # Dataloaders - Highly optimized for GPU
    # Increase num_workers for faster data loading (adjust based on CPU cores)
    cpu_count = os.cpu_count() or 4
    num_workers = min(8, cpu_count - 1) if cuda_available else 2  # Use more workers
    pin_memory = cuda_available
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch
        drop_last=False  # Keep all validation samples
    )
    
    print(f"DataLoader optimization: {num_workers} workers, prefetch_factor=4")
    
    # Create model
    model = create_model(model_name, n_classes=len(Config.EXPRESSIONS), pretrained=True)
    model = model.to(device)
    
    # Checkpoint management: Load existing weights if available
    checkpoint_path = os.path.join(Config.MODEL_SAVE_DIR, f'{save_name}_checkpoint.pth')
    start_epoch = 0
    best_val_f1_loaded = 0.0
    history_loaded = None
    
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"\n{'='*70}")
        print("CHECKPOINT FOUND - RESUMING TRAINING")
        print(f"{'='*70}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_f1_loaded = checkpoint.get('best_val_f1', 0.0)
            history_loaded = checkpoint.get('history', None)
            print(f"[OK] Loaded checkpoint from epoch {start_epoch}")
            print(f"[OK] Previous best F1 score: {best_val_f1_loaded:.4f}")
            print(f"[OK] Continuing training from epoch {start_epoch + 1}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            print(f"[INFO] Starting training from scratch\n")
            start_epoch = 0
            best_val_f1_loaded = 0.0
            history_loaded = None
    else:
        if resume_from_checkpoint:
            print(f"\n[INFO] No checkpoint found at {checkpoint_path}")
            print(f"[INFO] Starting training from scratch\n")
        else:
            print(f"\n[INFO] Checkpoint resumption disabled - starting fresh\n")
    
    # Verify model is on GPU
    if cuda_available:
        next_param_device = next(model.parameters()).device
        print(f"Model device: {next_param_device}")
        if next_param_device.type != 'cuda':
            print("WARNING: Model parameters are not on GPU! Training will be slow.")
        else:
            print(f"[OK] Model successfully moved to GPU: {torch.cuda.get_device_name(next_param_device.index)}")
    
    # Compile model for better performance (PyTorch 2.0+)
    # Note: Requires Triton, skip if not available
    model_compiled = False
    try:
        if hasattr(torch, 'compile') and cuda_available:
            try:
                import triton
                print("Compiling model for optimized GPU performance...")
                model = torch.compile(model, mode='reduce-overhead')
                print("[OK] Model compiled successfully")
                model_compiled = True
            except ImportError:
                print("Note: Triton not available, skipping model compilation (using standard mode)")
    except Exception as e:
        print(f"Note: Model compilation skipped, using standard mode")
    
    # GPU optimizations
    if cuda_available:
        # Enable cuDNN benchmark for faster convolutions (when input sizes don't change)
        torch.backends.cudnn.benchmark = True
        # Enable deterministic mode (optional - set to False for better performance)
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"cuDNN benchmark: Enabled (faster training)")
    
    # Loss and optimizer with enhanced regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Increased label smoothing
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Increased L2 regularization
    scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
    
    # Restore optimizer state if resuming from checkpoint
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[OK] Optimizer state restored")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"[OK] Scheduler state restored")
        except Exception as e:
            print(f"[WARNING] Could not restore optimizer/scheduler: {e}")
    
    # Training loop - restore previous best metrics and history if available
    best_val_f1 = best_val_f1_loaded
    best_val_acc = 0.0
    
    if history_loaded:
        history = history_loaded
        print(f"[OK] Training history restored with {len(history['train_loss'])} previous epochs\n")
    else:
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1_macro': []
        }
    
    print(f"\nStarting training...")
    print(f"{'='*70}\n")
    
    # Skip GPU test to save time - model is already verified on GPU
    
    # Adjust epoch range if resuming
    total_epochs = start_epoch + epochs
    for epoch in range(start_epoch, total_epochs):
        # Train
        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        
        # Sync GPU operations
        if cuda_available:
            torch.cuda.synchronize()
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, _ = validate_epoch(
            model, val_loader, criterion, device, use_amp
        )
        
        # Print GPU memory usage
        if cuda_available:
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        # Compute metrics
        val_metrics = compute_metrics(val_labels, val_preds, Config.EXPRESSIONS)
        val_f1_macro = val_metrics['f1_macro']
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{total_epochs}]")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1_macro:.4f}")
        
        # Save checkpoint after every epoch (for recovery)
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'history': history
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model (based on F1 score - best for imbalanced classes)
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_val_acc = val_acc
            model_path = os.path.join(Config.MODEL_SAVE_DIR, f'{save_name}.pth')
            torch.save(model.state_dict(), model_path)
            
            # Save best metrics
            metrics_path = os.path.join(Config.MODEL_SAVE_DIR, f'{save_name}_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)
            
            print(f"  [OK] Best model saved (F1: {val_f1_macro:.4f}, Acc: {val_acc:.2f}%)")
        
        print("-"*70)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Validation F1 (macro): {best_val_f1:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Plot training curves
    curves_path = os.path.join(Config.RESULTS_DIR, f'{save_name}_training_curves.png')
    plot_training_curves(history, curves_path)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(model_path))
    _, _, final_preds, final_labels, _ = validate_epoch(
        model, val_loader, criterion, device, use_amp
    )
    final_metrics = compute_metrics(final_labels, final_preds, Config.EXPRESSIONS)
    
    # Plot confusion matrix
    cm_path = os.path.join(Config.RESULTS_DIR, f'{save_name}_confusion_matrix.png')
    plot_confusion_matrix(
        np.array(final_metrics['confusion_matrix']), 
        Config.EXPRESSIONS, 
        cm_path
    )
    
    # Print final metrics
    print_metrics_report(final_metrics)
    
    # Cleanup GPU memory
    if cuda_available:
        torch.cuda.empty_cache()
        print(f"\nGPU memory cleared after training")
    
    return model, final_metrics, history

# ============================================================================
# MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """Evaluate and compare different models"""
    
    def __init__(self, test_data_dir, device='cuda'):
        self.test_data_dir = test_data_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.test_loader = None
        
    def load_test_data(self, batch_size=32):
        """Load test dataset"""
        test_dataset = ExpressionDataset(
            self.test_data_dir, 
            transform=val_transform
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        return self.test_loader
    
    def evaluate_model(self, model_path, model_name='efficientnet_b0'):
        """Evaluate a trained model on test set"""
        # Load model
        model = create_model(model_name, n_classes=len(Config.EXPRESSIONS), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_path}")
        print(f"{'='*70}")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, Config.EXPRESSIONS)
        
        # Print report
        print_metrics_report(metrics)
        
        return metrics, all_probs

# ============================================================================
# INFERENCE ENGINE (VIDEO MODE)
# ============================================================================

class InferenceEngine:
    """Handle real-time inference on video"""
    
    def __init__(self, model_path, model_name='efficientnet_b0', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, model_name)
        self.transform = inference_transform
        
        # Use YOLO for face detection if available, else fallback to Haar Cascade
        if YOLO_AVAILABLE:
            self.face_detector = YOLO('yolov8n.pt')  # Nano model for speed
            self.face_cascade = None
        else:
            self.face_detector = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_model(self, model_path, model_name):
        """Load trained model"""
        model = create_model(model_name, n_classes=len(Config.EXPRESSIONS), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        print(f"Model loaded: {model_path}")
        return model
    
    def predict(self, image, return_probs=False):
        """
        Predict expression from image
        
        Args:
            image: PIL Image or numpy array (BGR)
            return_probs: Return class probabilities
        
        Returns:
            expression: Predicted expression name
            probs: Class probabilities (if return_probs=True)
        """
        # Convert if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Transform and predict
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax().item()
            
        expression = Config.EXPRESSIONS[pred_idx]
        
        if return_probs:
            return expression, probs[0].cpu().numpy()
        return expression
    
    def detect_face(self, frame):
        """Detect faces in frame using YOLO or Haar Cascade"""
        if YOLO_AVAILABLE and self.face_detector is not None:
            # YOLO detection
            results = self.face_detector(frame, verbose=False, classes=[0])  # Class 0 = person
            faces = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if box.conf[0] > 0.5:  # Confidence threshold
                        faces.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
            faces = np.array(faces) if faces else np.array([])
        else:
            # Haar Cascade fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0, faces
    
    def is_good_pose(self, frame, faces):
        """Check if face is well-positioned"""
        if len(faces) == 0:
            return False, "No face detected"
        
        h, w = frame.shape[:2]
        x, y, fw, fh = faces[0]
        
        face_center_x = x + fw // 2
        face_center_y = y + fh // 2
        
        center_threshold = 0.4
        if (abs(face_center_x - w//2) / w < center_threshold and
            abs(face_center_y - h//2) / h < center_threshold and
            fw > 50 and fh > 50):
            return True, "Good pose detected!"
        
        return False, "Face not centered or too small"
    
    def process_video(self, video_source=0, save_dir='snapshots', 
                     frame_interval=3, usn='24ABCD233', person_num=1):
        """
        Process video feed with real-time expression detection
        
        Args:
            video_source: 0 for webcam, or path to video file
            save_dir: Directory to save snapshots
            frame_interval: Save frame every N seconds
            usn: User identifier
            person_num: Person number
        
        Returns:
            stats: Dictionary with predictions and statistics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        
        stats = {
            'predictions': [],
            'confidences': [],
            'files': [],
            'timestamps': []
        }
        
        frame_count = 0
        saved_count = 0
        last_extract_time = 0
        
        print(f"\n{'='*70}")
        print("VIDEO INFERENCE MODE")
        print(f"{'='*70}")
        print(f"Extracting frame every {frame_interval} seconds")
        print("Press 'q' to quit, 's' to save current frame manually")
        print(f"{'='*70}\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # Detect face
            has_face, faces = self.detect_face(frame)
            is_good, message = self.is_good_pose(frame, faces)
            
            # Draw face detection
            if has_face:
                for (x, y, w, h) in faces:
                    color = (0, 255, 0) if is_good else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, message, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Extract and predict at intervals
            if current_time - last_extract_time >= frame_interval:
                if is_good:
                    # Predict expression
                    expr, probs = self.predict(frame, return_probs=True)
                    confidence = probs.max()
                    
                    # Save snapshot
                    saved_count += 1
                    expr_code = Config.EXPR_CODES[expr]
                    filename = f"{usn}-{person_num:02d}-{expr_code}-{saved_count:02d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Record stats
                    stats['predictions'].append(expr)
                    stats['confidences'].append(float(confidence))
                    stats['files'].append(filename)
                    stats['timestamps'].append(current_time)
                    
                    print(f"[{current_time:.1f}s] [OK] {expr} (conf: {confidence:.3f}) -> {filename}")
                    last_extract_time = current_time
                else:
                    print(f"[{current_time:.1f}s] ✗ {message}")
                    last_extract_time = current_time
            
            # Display info
            cv2.putText(frame, f"Frame: {frame_count} | Saved: {saved_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Expression Detection - Press Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and has_face:
                # Manual save
                expr, probs = self.predict(frame, return_probs=True)
                saved_count += 1
                expr_code = Config.EXPR_CODES[expr]
                filename = f"{usn}-{person_num:02d}-{expr_code}-{saved_count:02d}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"Manually saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*70}")
        print("VIDEO PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total frames processed: {frame_count}")
        print(f"Snapshots saved: {saved_count}")
        print(f"Snapshots directory: {save_dir}/")
        
        return stats
    
    def process_folder(self, folder_path, output_dir=None):
        """
        Process all images in a folder with batch inference
        
        Args:
            folder_path: Path to folder containing images
            output_dir: Optional directory to save results (if None, prints only)
        
        Returns:
            stats: Dictionary with predictions and statistics
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # Get all image files
        image_files = []
        if os.path.isdir(folder_path):
            image_files = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions)
            ]
        else:
            print(f"Error: {folder_path} is not a valid directory")
            return None
        
        if len(image_files) == 0:
            print(f"No images found in {folder_path}")
            return None
        
        # Sort for consistent processing
        image_files = sorted(image_files)
        
        stats = {
            'predictions': [],
            'confidences': [],
            'files': [],
            'probs_all': []
        }
        
        print(f"\n{'='*70}")
        print("BATCH INFERENCE MODE")
        print(f"{'='*70}")
        print(f"Processing {len(image_files)} images from: {folder_path}")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Results will be saved to: {output_dir}")
        print(f"{'='*70}\n")
        
        # Process images in batches for efficiency
        batch_size = 32
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            batch_paths = []
            
            # Load batch
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    batch_paths.append(img_path)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # Batch transform
            batch_tensors = torch.stack([
                self.transform(img) for img in batch_images
            ]).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
            
            # Process results
            for j, (img_path, pred_idx, prob) in enumerate(zip(batch_paths, preds, probs)):
                expression = Config.EXPRESSIONS[pred_idx.item()]
                confidence = prob[pred_idx].item()
                
                stats['predictions'].append(expression)
                stats['confidences'].append(float(confidence))
                stats['files'].append(os.path.basename(img_path))
                stats['probs_all'].append(prob.cpu().numpy().tolist())
                
                # Save to output directory if specified
                if output_dir:
                    expr_code = Config.EXPR_CODES[expression]
                    filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{expr_code}_{confidence:.3f}.jpg"
                    output_path = os.path.join(output_dir, filename)
                    batch_images[j].save(output_path, 'JPEG')
                
                print(f"[{i+j+1}/{len(image_files)}] {os.path.basename(img_path):30s} -> {expression:12s} (conf: {confidence:.3f})")
        
        print(f"\n{'='*70}")
        print("BATCH INFERENCE COMPLETE")
        print(f"{'='*70}")
        print(f"Total images processed: {len(stats['predictions'])}")
        
        # Print distribution
        expr_counts = Counter(stats['predictions'])
        print(f"\nExpression distribution:")
        for expr in Config.EXPRESSIONS:
            count = expr_counts.get(expr, 0)
            pct = count / len(stats['predictions']) * 100 if stats['predictions'] else 0
            print(f"  {expr:<12}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\nAverage confidence: {np.mean(stats['confidences']):.3f}")
        print(f"Min confidence: {np.min(stats['confidences']):.3f}")
        print(f"Max confidence: {np.max(stats['confidences']):.3f}")
        
        # Save stats
        if output_dir:
            stats_path = os.path.join(output_dir, 'inference_stats.json')
            # Remove probs_all for JSON (too large)
            stats_to_save = {k: v for k, v in stats.items() if k != 'probs_all'}
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"\n[OK] Stats saved to: {stats_path}")
        
        print(f"{'='*70}\n")
        
        return stats

# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main execution with proper workflow separation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Expression Detection - Optimized Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train EfficientNet model (will use GPU if available)
  python expression-detection-optimized.py --mode train --data_dir dataset/train --epochs 20
  
  # Train with GPU requirement (exits if GPU not available)
  python expression-detection-optimized.py --mode train --data_dir dataset/train --epochs 20 --require_gpu
  
  # Evaluate trained model
  python expression-detection-optimized.py --mode evaluate --model_path models/best_model.pth --test_dir dataset/test
  
  # Batch inference on folder of images
  python expression-detection-optimized.py --mode inference --model_path models/best_model.pth --input_folder snapshots --output_folder inference_results
  
  # Run video inference (webcam)
  python expression-detection-optimized.py --mode inference --model_path models/best_model.pth --video_source 0
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate', 'inference'],
                       help='Execution mode')
    
    # Training arguments
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Dataset directory')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    parser.add_argument('--save_name', type=str, default='best_model',
                       help='Model save name')
    parser.add_argument('--require_gpu', action='store_true',
                       help='Exit if GPU is not available (for training mode)')
    
    # Evaluation arguments
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default='test_dataset',
                       help='Test dataset directory')
    
    # Inference arguments
    parser.add_argument('--video_source', default=0,
                       help='Video source (0 for webcam or video path)')
    parser.add_argument('--frame_interval', type=int, default=3,
                       help='Frame extraction interval (seconds)')
    parser.add_argument('--usn', type=str, default='24ABCD233',
                       help='User identifier')
    parser.add_argument('--person_num', type=int, default=1,
                       help='Person number')
    parser.add_argument('--save_dir', type=str, default='snapshots',
                       help='Snapshot save directory')
    parser.add_argument('--input_folder', type=str,
                       help='Folder containing images for batch inference')
    parser.add_argument('--output_folder', type=str,
                       help='Output folder for batch inference results')
    
    args = parser.parse_args()
    
    # Convert video_source to int if it's a digit
    if isinstance(args.video_source, str) and args.video_source.isdigit():
        args.video_source = int(args.video_source)
    
    # Check GPU requirement for training
    if args.mode == 'train' and args.require_gpu:
        if not torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("ERROR: GPU is required but not available!")
            print("=" * 70)
            print("To fix this:")
            print("  1. Check GPU drivers: nvidia-smi")
            print("  2. Install PyTorch with CUDA:")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("  3. Verify: python -c 'import torch; print(torch.cuda.is_available())'")
            print("=" * 70)
            exit(1)
        else:
            print("\n[OK] GPU requirement satisfied - GPU is available")
    
    # Execute based on mode
    if args.mode == 'train':
        model, metrics, history = train_model(
            data_dir=args.data_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_split=args.train_split,
            save_name=args.save_name,
            resume_from_checkpoint=True  # Always try to resume from checkpoint
        )
        
        print(f"\n[OK] Training completed successfully!")
        print(f"[OK] Model saved to: {Config.MODEL_SAVE_DIR}/{args.save_name}.pth")
        print(f"[OK] Results saved to: {Config.RESULTS_DIR}/")
    
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("ERROR: --model_path required for evaluation")
            return
        
        # Create evaluator
        evaluator = ModelEvaluator(args.test_dir)
        evaluator.load_test_data(batch_size=args.batch_size)
        
        # Evaluate model
        metrics, probs = evaluator.evaluate_model(
            args.model_path,
            args.model
        )
        
        # Save detailed report
        report_path = os.path.join(
            Config.RESULTS_DIR, 
            f'{args.save_name}_test_report.json'
        )
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[OK] Detailed report saved to: {report_path}")
    
    elif args.mode == 'inference':
        if not args.model_path:
            print("ERROR: --model_path required for inference")
            return
        
        # Create inference engine
        engine = InferenceEngine(
            model_path=args.model_path,
            model_name=args.model
        )
        
        # Check if batch inference on folder or video inference
        if args.input_folder:
            # Batch inference on folder
            stats = engine.process_folder(
                folder_path=args.input_folder,
                output_dir=args.output_folder
            )
        else:
            # Process video
            stats = engine.process_video(
                video_source=args.video_source,
                save_dir=args.save_dir,
                frame_interval=args.frame_interval,
                usn=args.usn,
                person_num=args.person_num
            )
            
            if stats and stats['predictions']:
                # Print statistics
                print(f"\n{'='*70}")
                print("INFERENCE STATISTICS")
                print(f"{'='*70}")
                print(f"Total predictions: {len(stats['predictions'])}")
                print(f"\nExpression distribution:")
                
                expr_counts = Counter(stats['predictions'])
                for expr in Config.EXPRESSIONS:
                    count = expr_counts.get(expr, 0)
                    pct = count / len(stats['predictions']) * 100 if stats['predictions'] else 0
                    print(f"  {expr:<12}: {count:3d} ({pct:5.1f}%)")
                
                print(f"\nAverage confidence: {np.mean(stats['confidences']):.3f}")
                
                # Save stats
                stats_path = os.path.join(args.save_dir, 'inference_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"\n[OK] Stats saved to: {stats_path}")
                print(f"{'='*70}")

if __name__ == '__main__':
    main()
