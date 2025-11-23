"""Phase 3a Warm-up Training Script

Trains classification heads of pretrained emotion models with frozen backbones.
Models: MobileNetV2, EfficientNet-B0, ResNet18, ResNet50

Usage:
    python train_phase3a.py --data_splits data_splits --epochs 20 --patience 5
    python train_phase3a.py --models mobilenet,resnet18 --epochs 15

Outputs:
 - Best checkpoints: results/checkpoints/<model>_warmup_best.pt
 - Logs CSV: results/logs/<model>_warmup.csv
 - Summary CSV: results/summary_phase3a.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.optim import Adam

from config import NUM_EPOCHS_WARMUP, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, USE_GPU, GPU_ID, DATA_SPLITS_DIR, EARLY_STOPPING_PATIENCE
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


def warmup_train_single(model_name: str, loaders, epochs: int, patience: int, results_dir: Path):
    print(f"\n=== Warm-up Training: {model_name} ===")

    device = get_device()
    model = create_model(model_name, num_classes=7, pretrained=True)

    # Freeze backbone
    model.freeze_backbone()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    trainer = Trainer(model=model, device=device, criterion=criterion, optimizer=optimizer, model_name=model_name, results_dir=results_dir)
    history = trainer.fit(loaders['train'], loaders['val'], epochs=epochs, patience=patience)
    test_loss, test_acc = trainer.evaluate_test(loaders['test'])

    print(f"[Test] {model_name}: Loss={test_loss:.4f} Acc={test_acc:.4f}")

    return {
        'model': model_name,
        'best_val_acc': history['best_val_acc'],
        'best_val_loss': history['best_val_loss'],
        'epochs_trained': history['epochs_trained'],
        'test_acc': test_acc,
        'test_loss': test_loss
    }


def append_summary(summary_path: Path, record: dict, header_written: bool):
    write_header = not header_written and not summary_path.exists()

    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(['model', 'epochs_trained', 'best_val_acc', 'best_val_loss', 'test_acc', 'test_loss'])

        writer.writerow([record['model'], record['epochs_trained'], f"{record['best_val_acc']:.4f}", f"{record['best_val_loss']:.4f}", f"{record['test_acc']:.4f}", f"{record['test_loss']:.4f}"])


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3a Warm-up Training')

    parser.add_argument('--data_splits', type=str, default=DATA_SPLITS_DIR, help='Path to data_splits root')
    parser.add_argument('--models', type=str, default=','.join(AVAILABLE_MODELS), help='Comma-separated list of models to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS_WARMUP, help='Max epochs for warm-up training')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Early stopping patience (default: 20 from config)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--no_weighted_sampler', action='store_true', help='Disable weighted sampler for class imbalance')

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print('='*70)
    print('PHASE 3a WARM-UP TRAINING')
    print('='*70)
    print(f"Device: {device}")
    print(f"Data splits: {args.data_splits}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}")
    print('='*70)

    # Build loaders once (shared across models)
    # Use CONSERVATIVE augmentation for warmup (frozen backbone)
    loaders = build_dataloaders(
        args.data_splits, 
        batch_size=args.batch_size, 
        use_weighted_sampler=not args.no_weighted_sampler, 
        pin_memory=device.type == 'cuda',
        augmentation_mode='warmup'  # Conservative augmentation for frozen backbone
    )

    models_to_train = [m.strip() for m in args.models.split(',') if m.strip() in AVAILABLE_MODELS]
    if not models_to_train:
        print('No valid models specified.')
        return 1

    results_dir = Path('../results')
    summary_path = results_dir / 'summary_phase3a.csv'
    results_dir.mkdir(exist_ok=True)

    for model_name in models_to_train:
        try:
            record = warmup_train_single(model_name, loaders, epochs=args.epochs, patience=args.patience, results_dir=results_dir)
            append_summary(summary_path, record, header_written=False)

        except Exception as e:
            print(f"[ERROR] Training failed for {model_name}: {e}")
            continue

    print('\nSummary written to', summary_path)
    print('Phase 3a complete.')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
