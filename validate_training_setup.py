"""Validation and Diagnostic Script for Training Setup

Tests to run BEFORE full training to ensure pipeline is working:
1. Initial loss check (~1.946 for 7 classes = -ln(1/7))
2. Overfitting test on small subset (should reach ~100% if model can learn)
3. Data normalization verification
4. Augmentation visualization (optional)

Usage:
    python validate_training_setup.py --model mobilenet
    python validate_training_setup.py --model resnet18 --overfit_epochs 50
"""

import argparse
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

from config import (
    NUM_CLASSES, LEARNING_RATE, BATCH_SIZE, 
    USE_GPU, GPU_ID, DATA_SPLITS_DIR
)
from models import create_model
from utils.data_loader import build_dataloaders, IMAGENET_MEAN, IMAGENET_STD


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def test_initial_loss(model, loader, device):
    """Test 1: Initial loss should be ~1.946 for 7 classes"""
    print("\n" + "="*70)
    print("TEST 1: Initial Loss Validation")
    print("="*70)
    
    expected_loss = -math.log(1.0 / NUM_CLASSES)
    print(f"Expected initial loss (random guessing): {expected_loss:.4f}")
    print(f"Acceptable range: {expected_loss-0.2:.4f} to {expected_loss+0.2:.4f}")
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if i >= 10:  # Test on first 10 batches
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
    
    avg_loss = total_loss / total_samples
    print(f"Actual initial loss: {avg_loss:.4f}")
    
    if abs(avg_loss - expected_loss) < 0.3:
        print("✅ PASS: Initial loss is within expected range")
        print("   → Model initialization and normalization are correct")
    else:
        print("❌ FAIL: Initial loss is outside expected range")
        print("   → Check ImageNet normalization or model initialization")
    
    return avg_loss


def test_overfitting_capability(model, train_loader, device, epochs=50):
    """Test 2: Model should overfit to 100% on small subset"""
    print("\n" + "="*70)
    print("TEST 2: Overfitting Capability Test")
    print("="*70)
    print(f"Training on first batch for {epochs} epochs...")
    print("Goal: Reach ~100% accuracy (proves model can learn)")
    
    # Get single batch
    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=LEARNING_RATE * 10, weight_decay=0)  # Higher LR, no regularization
    
    print(f"\nBatch size: {targets.size(0)} samples")
    print(f"Optimized learning rate: {LEARNING_RATE * 10:.2e} (10x normal for overfitting test)")
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%")
        
        # Early stop if perfect
        if acc >= 0.99:
            print(f"\n✅ PASS: Achieved {acc*100:.1f}% accuracy at epoch {epoch}")
            print("   → Model architecture and optimizer are working correctly")
            return True
    
    final_acc = acc * 100
    if final_acc >= 95:
        print(f"\n✅ PASS: Achieved {final_acc:.1f}% accuracy")
        print("   → Model can learn, might need more epochs for 100%")
        return True
    else:
        print(f"\n❌ FAIL: Only achieved {final_acc:.1f}% accuracy after {epochs} epochs")
        print("   → Model may have issues with architecture, optimizer, or gradients")
        return False


def test_data_normalization():
    """Test 3: Verify ImageNet normalization constants"""
    print("\n" + "="*70)
    print("TEST 3: Data Normalization Verification")
    print("="*70)
    
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    
    print(f"Expected ImageNet mean: {expected_mean}")
    print(f"Actual mean in loader:  {IMAGENET_MEAN}")
    
    print(f"Expected ImageNet std:  {expected_std}")
    print(f"Actual std in loader:   {IMAGENET_STD}")
    
    if IMAGENET_MEAN == expected_mean and IMAGENET_STD == expected_std:
        print("✅ PASS: ImageNet normalization is correct")
    else:
        print("❌ FAIL: ImageNet normalization is incorrect")
        print("   → This will cause pretrained models to fail!")
    
    return IMAGENET_MEAN == expected_mean and IMAGENET_STD == expected_std


def main():
    parser = argparse.ArgumentParser(description='Validate Training Setup')
    parser.add_argument('--model', type=str, default='mobilenet', 
                        choices=['mobilenet', 'efficientnet', 'resnet18', 'resnet50'],
                        help='Model to test')
    parser.add_argument('--overfit_epochs', type=int, default=50,
                        help='Epochs for overfitting test')
    parser.add_argument('--skip_overfit', action='store_true',
                        help='Skip overfitting test (saves time)')
    args = parser.parse_args()
    
    device = get_device()
    print("="*70)
    print("TRAINING SETUP VALIDATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load model
    model = create_model(args.model, num_classes=NUM_CLASSES, pretrained=True)
    model.freeze_backbone()
    model = model.to(device)
    
    # Load data
    loaders = build_dataloaders(DATA_SPLITS_DIR, batch_size=BATCH_SIZE, 
                                use_weighted_sampler=False, pin_memory=device.type == 'cuda')
    
    # Run tests
    results = {}
    
    # Test 1: Initial loss
    results['initial_loss'] = test_initial_loss(model, loaders['val'], device)
    
    # Test 2: Overfitting capability
    if not args.skip_overfit:
        results['overfitting'] = test_overfitting_capability(model, loaders['train'], 
                                                             device, args.overfit_epochs)
    
    # Test 3: Normalization
    results['normalization'] = test_data_normalization()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if not args.skip_overfit:
        all_passed = (
            abs(results['initial_loss'] - (-math.log(1/NUM_CLASSES))) < 0.3 and
            results['overfitting'] and
            results['normalization']
        )
    else:
        all_passed = (
            abs(results['initial_loss'] - (-math.log(1/NUM_CLASSES))) < 0.3 and
            results['normalization']
        )
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYour training setup is ready. Expected results:")
        print("  • Warmup phase (frozen backbone):   55-70% validation accuracy")
        print("  • Fine-tuning phase (unfrozen):     75-85% validation accuracy")
        print("\nRun training with:")
        print("  python train_phase3a.py --epochs 20")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running full training.")
        print("Common fixes:")
        print("  • Initial loss too high: Check ImageNet normalization")
        print("  • Can't overfit: Check model architecture or learning rate")
        print("  • Wrong normalization: Update IMAGENET_MEAN/STD in data_loader.py")
    
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
