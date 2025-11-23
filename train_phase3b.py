"""Phase 3b Fine-Tuning Training Script

Fine-tunes emotion models by unfreezing backbone layers after warm-up.
Models: MobileNetV2, EfficientNet-B0, ResNet18, ResNet50

Loads best warm-up checkpoint and continues training with unfrozen layers.

Usage:
    python train_phase3b.py --data_splits data_splits --epochs 20 --unfreeze_layers 2
    python train_phase3b.py --models mobilenet,resnet18 --epochs 15 --unfreeze_layers 3
    python train_phase3b.py --models efficientnet --unfreeze_layers -1  # Full unfreeze

Outputs:
 - Best checkpoints: results/checkpoints/<model>_finetune_best.pt
 - Logs CSV: results/logs/<model>_finetune.csv
 - Summary CSV: results/summary_phase3b.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.optim import Adam

from config import NUM_EPOCHS_FINETUNE, FINE_TUNE_LR, WEIGHT_DECAY, BATCH_SIZE, USE_GPU, GPU_ID, DATA_SPLITS_DIR, FINETUNE_PATIENCE
from src.models import create_model
from utils.data_loader import build_dataloaders
from utils.trainer import Trainer


AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass

        return torch.device(f'cuda:{GPU_ID}')
    
    return torch.device('cpu')


def finetune_train_single(model_name: str, loaders, epochs: int, patience: int, num_unfreeze: int, results_dir: Path):
    """Fine-tune a single model with unfrozen backbone layers.
    
    Args:
        model_name: Name of model architecture ('mobilenet', 'efficientnet', 'resnet18', 'resnet50')
        loaders: Dict with 'train', 'val', 'test' DataLoaders
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        num_unfreeze: Number of backbone layers to unfreeze (-1 = all)
        results_dir: Directory for saving checkpoints and logs
        
    Returns:
        Dict with training metrics (model, val_acc, test_acc, etc.)
    """
    print(f"\n=== Fine-Tuning Training: {model_name} ===")

    device = get_device()
    model = create_model(model_name, num_classes=7, pretrained=True)

    # Load warm-up checkpoint
    checkpoint_path = results_dir / 'checkpoints' / f'{model_name}_warmup_best.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-up checkpoint not found: {checkpoint_path}\nRun train_phase3a.py first.")
    
    print(f"Loading warm-up checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    warmup_val_acc = checkpoint.get('val_acc', 0.0)
    warmup_val_loss = checkpoint.get('val_loss', 0.0)
    warmup_epoch = checkpoint.get('epoch', 0)
    print(f"Warm-up best checkpoint: Epoch {warmup_epoch}, Val Acc: {warmup_val_acc:.4f}, Val Loss: {warmup_val_loss:.4f}")

    # Unfreeze backbone layers
    if num_unfreeze == -1:
        print("Unfreezing ALL backbone layers (full fine-tuning)")
        model.unfreeze_backbone(num_layers=-1)
    else:
        print(f"Unfreezing last {num_unfreeze} backbone layer group(s)")
        model.unfreeze_backbone(num_layers=num_unfreeze)
    
    # Verify parameter freezing status
    total_params = sum(1 for _ in model.parameters())
    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params
    print(f"Parameters: {trainable_params}/{total_params} trainable, {frozen_params} frozen")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Optimizer only for trainable parameters
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=FINE_TUNE_LR, 
        weight_decay=WEIGHT_DECAY
    )

    # Trainer with phase='finetune', no warmup scheduler
    trainer = Trainer(
        model=model, 
        device=device, 
        criterion=criterion, 
        optimizer=optimizer, 
        model_name=model_name, 
        results_dir=results_dir,
        phase='finetune',
        use_warmup=False,
        grad_clip_max_norm=1.0
    )
    
    history = trainer.fit(loaders['train'], loaders['val'], epochs=epochs, patience=patience)
    test_loss, test_acc = trainer.evaluate_test(loaders['test'])

    print(f"[Test] {model_name}: Loss={test_loss:.4f} Acc={test_acc:.4f}")
    print(f"Improvement: {warmup_val_acc:.4f} (warmup) → {history['best_val_acc']:.4f} (finetune) = +{history['best_val_acc'] - warmup_val_acc:.4f}")

    return {
        'model': model_name,
        'warmup_val_acc': warmup_val_acc,
        'best_val_acc': history['best_val_acc'],
        'best_val_loss': history['best_val_loss'],
        'epochs_trained': history['epochs_trained'],
        'test_acc': test_acc,
        'test_loss': test_loss,
        'num_unfreeze': num_unfreeze
    }


def append_summary(summary_path: Path, record: dict, header_written: bool):
    write_header = not header_written and not summary_path.exists()

    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                'model', 'num_unfreeze', 'warmup_val_acc', 'epochs_trained', 
                'best_val_acc', 'best_val_loss', 'test_acc', 'test_loss'
            ])

        writer.writerow([
            record['model'], 
            record['num_unfreeze'],
            f"{record['warmup_val_acc']:.4f}",
            record['epochs_trained'], 
            f"{record['best_val_acc']:.4f}", 
            f"{record['best_val_loss']:.4f}", 
            f"{record['test_acc']:.4f}", 
            f"{record['test_loss']:.4f}"
        ])


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3b Fine-Tuning Training')

    parser.add_argument('--data_splits', type=str, default=DATA_SPLITS_DIR, help='Path to data_splits root')
    parser.add_argument('--models', type=str, default=','.join(AVAILABLE_MODELS), help='Comma-separated list of models to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS_FINETUNE, help='Max epochs for fine-tuning')
    parser.add_argument('--patience', type=int, default=FINETUNE_PATIENCE, help='Early stopping patience (default: 5 from config)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--no_weighted_sampler', action='store_true', help='Disable weighted sampler for class imbalance')
    parser.add_argument('--unfreeze_layers', type=int, default=2, help='Number of backbone layers to unfreeze (2=last 2 groups, -1=all layers)')

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print('='*70)
    print('PHASE 3b FINE-TUNING TRAINING')
    print('='*70)
    print(f"Device: {device}")
    print(f"Data splits: {args.data_splits}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}")
    print(f"Unfreeze layers: {args.unfreeze_layers}")
    print(f"Fine-tune LR: {FINE_TUNE_LR} | Weight decay: {WEIGHT_DECAY}")
    print('='*70)

    # Build loaders once (shared across models)
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

    results_dir = Path('../results')
    summary_path = results_dir / 'summary_phase3b.csv'
    results_dir.mkdir(exist_ok=True)

    for model_name in models_to_train:
        try:
            record = finetune_train_single(
                model_name, 
                loaders, 
                epochs=args.epochs, 
                patience=args.patience, 
                num_unfreeze=args.unfreeze_layers,
                results_dir=results_dir
            )
            append_summary(summary_path, record, header_written=False)

        except Exception as e:
            print(f"[ERROR] Training failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print('\nSummary written to', summary_path)
    print('Phase 3b complete.')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
