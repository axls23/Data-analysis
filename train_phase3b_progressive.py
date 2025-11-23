"""Phase 3b Progressive Fine-Tuning Script (Stage 2 - Deep Fine-tuning)

Advanced fine-tuning with:
- Model-specific unfreezing strategies
- Cosine annealing learning rate schedule
- Label smoothing regularization
- Overfitting detection and early stopping
- Enhanced monitoring and visualization

This script implements "Stage 2" progressive fine-tuning:
- MobileNet: Unfreeze last 4 inverted residual blocks
- EfficientNet: Unfreeze last 5 MBConv blocks
- ResNet18: Unfreeze layer3 + layer4 (half the network)
- ResNet50: Unfreeze layer4 + last 2 blocks of layer3

Usage:
    python train_phase3b_progressive.py --epochs 20
    python train_phase3b_progressive.py --models mobilenet,resnet50 --epochs 15
    python train_phase3b_progressive.py --models resnet18 --use_label_smoothing --plot_curves

Outputs:
    - Best checkpoints: results/checkpoints/<model>_finetune_progressive_best.pt
    - Logs CSV: results/logs/<model>_finetune_progressive.csv
    - Summary CSV: results/summary_phase3b_progressive.csv
    - Training curves: results/curves/<model>_finetune_progressive_curves.png (if --plot_curves)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    NUM_EPOCHS_FINETUNE, FINE_TUNE_LR, BATCH_SIZE, 
    USE_GPU, GPU_ID, DATA_SPLITS_DIR, FINETUNE_PATIENCE,
    MODEL_SPECIFIC_PARAMS, LABEL_SMOOTHING, COSINE_T_MAX, COSINE_ETA_MIN
)

from models import create_model
from utils.data_loader import build_dataloaders
from utils.losses import LabelSmoothingCrossEntropy
from utils.monitoring import plot_training_curves, print_overfitting_report


AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def progressive_unfreeze_model(model_name: str, model: nn.Module) -> tuple[int, int]:
    """Apply model-specific progressive unfreezing strategy for Stage 2.
    
    Args:
        model_name: Name of model ('mobilenet', 'efficientnet', 'resnet18', 'resnet50')
        model: Model instance
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    params = MODEL_SPECIFIC_PARAMS[model_name]
    
    if model_name == 'mobilenet':
        # Unfreeze last N inverted residual blocks
        num_blocks = params['unfreeze_blocks_stage2']
        model.unfreeze_backbone(num_layers=num_blocks)
        print(f"[INFO] Unfroze last {num_blocks} inverted residual blocks of MobileNetV2")
    
    elif model_name == 'efficientnet':
        # Unfreeze last N MBConv blocks
        num_blocks = params['unfreeze_blocks_stage2']
        model.unfreeze_backbone(num_layers=num_blocks)
        print(f"[INFO] Unfroze last {num_blocks} MBConv blocks of EfficientNet-B0")
    
    elif model_name == 'resnet18':
        # Unfreeze specified layer groups (e.g., [3, 4] for layer3 and layer4)
        unfreeze_layers = params['unfreeze_layers_stage2']
        
        # First freeze all
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze specified layers
        for layer_idx in unfreeze_layers:
            layer = getattr(model.backbone, f'layer{layer_idx}')
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"[INFO] Unfroze ResNet18 layers: {', '.join([f'layer{i}' for i in unfreeze_layers])}")
    
    elif model_name == 'resnet50':
        # Unfreeze layer4 fully + last N blocks of layer3
        # First freeze all
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 completely
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Unfreeze last N blocks of layer3
        num_blocks_layer3 = params.get('unfreeze_blocks_layer3', 2)
        total_blocks_layer3 = len(model.backbone.layer3)
        start_idx = total_blocks_layer3 - num_blocks_layer3
        
        for i in range(start_idx, total_blocks_layer3):
            for param in model.backbone.layer3[i].parameters():
                param.requires_grad = True
        
        print(f"[INFO] Unfroze ResNet50 layer4 + last {num_blocks_layer3} blocks of layer3")
    
    # Always ensure classifier is unfrozen
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Count parameters
    total_params = sum(1 for _ in model.parameters())
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"[INFO] Parameters: {trainable_params}/{total_params} trainable, {total_params - trainable_params} frozen")
    
    return total_params, trainable_params


def progressive_finetune_single(
    model_name: str, 
    loaders, 
    epochs: int, 
    patience: int, 
    results_dir: Path,
    use_label_smoothing: bool = True,
    use_cosine_lr: bool = True,
    plot_curves: bool = False,
    resume: bool = False
) -> dict:
    """Progressive fine-tuning for a single model (Stage 2).
    
    Args:
        model_name: Model architecture name
        loaders: DataLoader dict with 'train', 'val', 'test'
        epochs: Maximum training epochs
        patience: Early stopping patience
        results_dir: Directory for outputs
        use_label_smoothing: Use label smoothing loss
        use_cosine_lr: Use cosine annealing LR schedule
        plot_curves: Generate training curve plots
        resume: Resume from previous progressive checkpoint
    
    Returns:
        Dict with training metrics
    """
    print(f"\n{'='*70}")
    print(f"PROGRESSIVE FINE-TUNING (STAGE 2): {model_name.upper()}")
    print(f"{'='*70}")
    
    device = get_device()
    
    # Get model-specific hyperparameters
    model_params = MODEL_SPECIFIC_PARAMS[model_name]
    dropout = model_params['dropout']
    weight_decay = model_params['weight_decay']
    lr_multiplier = model_params.get('fine_tune_lr_multiplier', 1.0)
    learning_rate = FINE_TUNE_LR * lr_multiplier
    
    print(f"[HYPERPARAMS] Dropout: {dropout}, Weight Decay: {weight_decay}, LR: {learning_rate:.2e}")
    
    # Create model with model-specific dropout
    model = create_model(model_name, num_classes=7, pretrained=True, dropout=dropout)
    
    # Determine checkpoint to load
    warmup_checkpoint_path = results_dir / 'checkpoints' / f'{model_name}_warmup_best.pt'
    progressive_checkpoint_path = results_dir / 'checkpoints' / f'{model_name}_finetune_progressive_best.pt'
    
    start_epoch = 1
    checkpoint_path = None
    
    if resume and progressive_checkpoint_path.exists():
        print(f"[RESUME] Found progressive checkpoint: {progressive_checkpoint_path}")
        checkpoint_path = progressive_checkpoint_path
    elif warmup_checkpoint_path.exists():
        print(f"[CHECKPOINT] Loading from warmup: {warmup_checkpoint_path}")
        checkpoint_path = warmup_checkpoint_path
    else:
        # Try Phase 3a checkpoint as fallback
        fallback_path = results_dir / 'checkpoints' / f'{model_name}_finetune_best.pt'
        if fallback_path.exists():
             print(f"[FALLBACK] Loading from Phase 3a: {fallback_path}")
             checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name}. "
                f"Run train_phase3a.py first."
            )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    warmup_val_acc = checkpoint.get('val_acc', 0.0)
    if resume and progressive_checkpoint_path.exists():
         print(f"[RESUME] Resuming from val_acc: {warmup_val_acc:.4f}")
    else:
         print(f"[BASELINE] Warmup validation accuracy: {warmup_val_acc:.4f}")
    
    # Apply progressive unfreezing (Stage 2)
    total_params, trainable_params = progressive_unfreeze_model(model_name, model)
    
    model = model.to(device)
    
    # Loss function
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        print(f"[LOSS] Using label smoothing with epsilon={LABEL_SMOOTHING}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"[LOSS] Using standard cross-entropy")
    
    # Optimizer (only trainable parameters, model-specific weight decay)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (Cosine Annealing)
    if use_cosine_lr:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=COSINE_T_MAX,
            eta_min=COSINE_ETA_MIN
        )
        print(f"[SCHEDULER] Cosine annealing: T_max={COSINE_T_MAX}, eta_min={COSINE_ETA_MIN:.2e}")
    else:
        scheduler = None
    
    # Training loop with early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Logging setup
    log_dir = results_dir / 'logs'
    checkpoint_dir = results_dir / 'checkpoints'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f'{model_name}_finetune_progressive.csv'
    best_checkpoint_path = checkpoint_dir / f'{model_name}_finetune_progressive_best.pt'
    
    # Initialize CSV
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    print(f"\n{'='*70}")
    print(f"TRAINING STARTED")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in loaders['train']:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in loaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * targets.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", 
                           f"{val_loss:.6f}", f"{val_acc:.4f}", f"{current_lr:.2e}"])
        
        print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
        
        # Check for improvement
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            improved = True
            
            # Save best checkpoint
            torch.save({
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'epoch': epoch,
                'optimizer_state': optimizer.state_dict()
            }, best_checkpoint_path)
        else:
            epochs_no_improve += 1
        
        if improved:
            print(f"  ✓ New best model (val_acc={best_val_acc:.4f})")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] No improvement for {patience} epochs. Stopping.")
            break
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}\n")
    
    # Test evaluation
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in loaders['test']:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * targets.size(0)
            test_correct += (outputs.argmax(1) == targets).sum().item()
            test_total += targets.size(0)
    
    test_loss /= test_total
    test_acc = test_correct / test_total
    
    print(f"[TEST RESULTS] {model_name}")
    print(f"  Warmup baseline:     {warmup_val_acc:.4f}")
    print(f"  Best validation:     {best_val_acc:.4f}")
    print(f"  Test accuracy:       {test_acc:.4f}")
    print(f"  Improvement:         +{(test_acc - warmup_val_acc):.4f} ({(test_acc - warmup_val_acc)*100:.2f}%)")
    
    # Plot training curves
    if plot_curves:
        try:
            plot_training_curves(log_path, show_lr=use_cosine_lr)
            curves_path = log_path.parents[1] / 'curves' / f"{log_path.stem}_curves.png"
            print(f"[INFO] Training curves saved to: {curves_path}")
        except Exception as e:
            print(f"[WARNING] Could not plot training curves: {e}")
    
    # Overfitting analysis
    try:
        print_overfitting_report(log_path)
    except Exception as e:
        print(f"[WARNING] Could not generate overfitting report: {e}")
    
    return {
        'model': model_name,
        'warmup_val_acc': warmup_val_acc,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'trainable_params': trainable_params,
        'total_params': total_params
    }


def append_summary(summary_path: Path, record: dict, header_written: bool):
    write_header = not header_written and not summary_path.exists()
    
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow([
                'model', 'trainable_params', 'total_params', 'warmup_val_acc',
                'epochs_trained', 'best_val_acc', 'best_val_loss', 'test_acc', 'test_loss'
            ])
        
        writer.writerow([
            record['model'],
            record['trainable_params'],
            record['total_params'],
            f"{record['warmup_val_acc']:.4f}",
            record['epochs_trained'],
            f"{record['best_val_acc']:.4f}",
            f"{record['best_val_loss']:.4f}",
            f"{record['test_acc']:.4f}",
            f"{record['test_loss']:.4f}"
        ])


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3b Progressive Fine-Tuning (Stage 2)')
    
    parser.add_argument('--data_splits', type=str, default=DATA_SPLITS_DIR,
                       help='Path to data_splits root')
    parser.add_argument('--models', type=str, default=','.join(AVAILABLE_MODELS),
                       help='Comma-separated list of models to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS_FINETUNE,
                       help='Max epochs for fine-tuning')
    parser.add_argument('--patience', type=int, default=FINETUNE_PATIENCE,
                       help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--no_weighted_sampler', action='store_true',
                       help='Disable weighted sampler')
    parser.add_argument('--no_label_smoothing', action='store_true',
                       help='Disable label smoothing (use standard cross-entropy)')
    parser.add_argument('--no_cosine_lr', action='store_true',
                       help='Disable cosine annealing LR schedule')
    parser.add_argument('--plot_curves', action='store_true',
                       help='Generate training curve plots')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous progressive fine-tuning checkpoint if available')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    
    print('='*70)
    print('PHASE 3b PROGRESSIVE FINE-TUNING (STAGE 2)')
    print('='*70)
    print(f"Device: {device}")
    print(f"Data splits: {args.data_splits}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}")
    print(f"Label smoothing: {not args.no_label_smoothing}")
    print(f"Cosine LR schedule: {not args.no_cosine_lr}")
    print('='*70)
    
    # Build dataloaders
    loaders = build_dataloaders(
        args.data_splits,
        batch_size=args.batch_size,
        use_weighted_sampler=not args.no_weighted_sampler,
        pin_memory=device.type == 'cuda'
    )
    
    models_to_train = [m.strip() for m in args.models.split(',') if m.strip() in AVAILABLE_MODELS]
    
    if not models_to_train:
        print('No valid models specified.')
        return 1
    
    results_dir = Path('results')
    summary_path = results_dir / 'summary_phase3b_progressive.csv'
    results_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_train:
        try:
            record = progressive_finetune_single(
                model_name,
                loaders,
                epochs=args.epochs,
                patience=args.patience,
                results_dir=results_dir,
                use_label_smoothing=not args.no_label_smoothing,
                use_cosine_lr=not args.no_cosine_lr,
                plot_curves=args.plot_curves,
                resume=args.resume
            )
            append_summary(summary_path, record, header_written=False)
        
        except Exception as e:
            print(f"[ERROR] Training failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"ALL TRAINING COMPLETE")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
