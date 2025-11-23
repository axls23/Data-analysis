"""Phase 3c Deep Fine-Tuning Script (Stage 3 - Full Unfreezing)

Final stage of progressive fine-tuning:
- Unfreezes ENTIRE backbone for maximum adaptability
- Uses lower learning rate (1e-5) to preserve learned features
- Continues from Stage 2 (Progressive) checkpoints
- Uses moderate-aggressive augmentation (same as Phase 3b)

Usage:
    python train_phase3c_deep.py --epochs 20
    python train_phase3c_deep.py --models resnet18,resnet50 --epochs 20 --plot_curves

Outputs:
    - Best checkpoints: results/checkpoints/<model>_finetune_deep_best.pt
    - Logs CSV: results/logs/<model>_finetune_deep.csv
    - Summary CSV: results/summary_phase3c_deep.csv
    - Training curves: results/curves/<model>_finetune_deep_curves.png (if --plot_curves)
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
    NUM_EPOCHS_FINETUNE, BATCH_SIZE, 
    USE_GPU, GPU_ID, DATA_SPLITS_DIR, FINETUNE_PATIENCE,
    MODEL_SPECIFIC_PARAMS, LABEL_SMOOTHING, COSINE_T_MAX, COSINE_ETA_MIN,
    FINE_TUNE_LR
)
from src.models import create_model
from utils.data_loader import build_dataloaders
from utils.losses import LabelSmoothingCrossEntropy
from utils.monitoring import plot_training_curves, print_overfitting_report

# Stage 3 specific constants
# DEEP_FINETUNE_LR = 1e-5  # Lower LR for full network fine-tuning
AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def deep_finetune_single(
    model_name: str, 
    loaders, 
    epochs: int, 
    patience: int, 
    results_dir: Path,
    use_label_smoothing: bool = True,
    use_cosine_lr: bool = True,
    plot_curves: bool = False
) -> dict:
    """Deep fine-tuning for a single model (Stage 3)."""
    
    print(f"\n{'='*70}")
    print(f"DEEP FINE-TUNING (STAGE 3): {model_name.upper()}")
    print(f"{'='*70}")
    
    device = get_device()
    
    # Get model-specific hyperparameters
    model_params = MODEL_SPECIFIC_PARAMS[model_name]
    dropout = model_params['dropout']
    weight_decay = model_params['weight_decay']
    
    # Use FINE_TUNE_LR from config (3e-5)
    learning_rate = FINE_TUNE_LR
    
    print(f"[HYPERPARAMS] Dropout: {dropout}, Weight Decay: {weight_decay}, LR: {learning_rate:.2e}")
    
    # Create model
    model = create_model(model_name, num_classes=7, pretrained=True, dropout=dropout)
    
    # Load Stage 2 checkpoint (Progressive)
    stage2_checkpoint_path = results_dir / 'checkpoints' / f'{model_name}_finetune_progressive_best.pt'
    
    if not stage2_checkpoint_path.exists():
        raise FileNotFoundError(
            f"No Stage 2 checkpoint found at {stage2_checkpoint_path}. "
            f"Run train_phase3b_progressive.py first."
        )
    
    print(f"[CHECKPOINT] Loading from Stage 2: {stage2_checkpoint_path}")
    checkpoint = torch.load(stage2_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    baseline_val_acc = checkpoint.get('val_acc', 0.0)
    print(f"[BASELINE] Stage 2 validation accuracy: {baseline_val_acc:.4f}")
    
    # Unfreeze EVERYTHING (Stage 3)
    model.unfreeze_backbone(num_layers=-1)
    
    # Always ensure classifier is unfrozen
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # Count parameters
    total_params = sum(1 for _ in model.parameters())
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {trainable_params}/{total_params} trainable (100% unfrozen)")
    
    # Loss function
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        print(f"[LOSS] Using label smoothing with epsilon={LABEL_SMOOTHING}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"[LOSS] Using standard cross-entropy")
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    if use_cosine_lr:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=COSINE_T_MAX,
            eta_min=COSINE_ETA_MIN
        )
        print(f"[SCHEDULER] Cosine annealing: T_max={COSINE_T_MAX}, eta_min={COSINE_ETA_MIN:.2e}")
    else:
        scheduler = None
    
    # Training loop
    best_val_acc = baseline_val_acc  # Start tracking from previous best
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Logging setup
    log_dir = results_dir / 'logs'
    checkpoint_dir = results_dir / 'checkpoints'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f'{model_name}_finetune_deep.csv'
    best_checkpoint_path = checkpoint_dir / f'{model_name}_finetune_deep_best.pt'
    
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
    if best_checkpoint_path.exists():
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
    print(f"  Stage 2 baseline:    {baseline_val_acc:.4f}")
    print(f"  Best validation:     {best_val_acc:.4f}")
    print(f"  Test accuracy:       {test_acc:.4f}")
    print(f"  Improvement:         +{(test_acc - baseline_val_acc):.4f} ({(test_acc - baseline_val_acc)*100:.2f}%)")
    
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
        'baseline_val_acc': baseline_val_acc,
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
                'model', 'trainable_params', 'total_params', 'baseline_val_acc',
                'epochs_trained', 'best_val_acc', 'best_val_loss', 'test_acc', 'test_loss'
            ])
        
        writer.writerow([
            record['model'],
            record['trainable_params'],
            record['total_params'],
            f"{record['baseline_val_acc']:.4f}",
            record['epochs_trained'],
            f"{record['best_val_acc']:.4f}",
            f"{record['best_val_loss']:.4f}",
            f"{record['test_acc']:.4f}",
            f"{record['test_loss']:.4f}"
        ])


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3c Deep Fine-Tuning (Stage 3)')
    
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    
    print('='*70)
    print('PHASE 3c DEEP FINE-TUNING (STAGE 3)')
    print('='*70)
    print(f"Device: {device}")
    print(f"Data splits: {args.data_splits}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}")
    print(f"Label smoothing: {not args.no_label_smoothing}")
    print(f"Cosine LR schedule: {not args.no_cosine_lr}")
    print('='*70)
    
    # Build dataloaders with FINETUNE augmentation (moderate-aggressive)
    loaders = build_dataloaders(
        args.data_splits,
        batch_size=args.batch_size,
        use_weighted_sampler=not args.no_weighted_sampler,
        pin_memory=device.type == 'cuda',
        augmentation_mode='finetune'
    )
    
    models_to_train = [m.strip() for m in args.models.split(',') if m.strip() in AVAILABLE_MODELS]
    
    if not models_to_train:
        print('No valid models specified.')
        return 1
    
    results_dir = Path('../results')
    summary_path = results_dir / 'summary_phase3c_deep.csv'
    results_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_train:
        try:
            record = deep_finetune_single(
                model_name,
                loaders,
                epochs=args.epochs,
                patience=args.patience,
                results_dir=results_dir,
                use_label_smoothing=not args.no_label_smoothing,
                use_cosine_lr=not args.no_cosine_lr,
                plot_curves=args.plot_curves
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
