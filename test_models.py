"""
Model Testing and Comparison Script
=====================================

Test all implemented models to verify they work correctly and compare their characteristics.

Usage:
    python test_models.py
    python test_models.py --model mobilenet
    python test_models.py --all
"""

import torch
import argparse
from models import create_model, list_available_models
from config import NUM_CLASSES, INPUT_SIZE


def get_device_info():
    """Get device information for GPU/CPU usage"""
    info = {
        'device': 'cpu',
        'name': 'CPU',
        'gpu_available': False,
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0
    }
    
    if torch.cuda.is_available():
        info['gpu_available'] = True
        info['device_count'] = torch.cuda.device_count()
        
        # Prefer NVIDIA GPU (device index 0)
        try:
            torch.cuda.set_device(0)
            device_name = torch.cuda.get_device_name(0)
            
            # Check if it's NVIDIA GPU
            if 'nvidia' in device_name.lower() or 'tesla' in device_name.lower() or 'geforce' in device_name.lower():
                info['device'] = 'cuda:0'
                info['name'] = f"NVIDIA GPU - {device_name}"
            else:
                # Other GPU, still use it
                info['device'] = 'cuda:0'
                info['name'] = f"GPU - {device_name}"
        except Exception as e:
            print(f"[WARNING] Error detecting GPU: {e}")
            info['name'] = "CPU (GPU detection failed)"
    
    return info


def test_model(model_name, device, verbose=True):
    """
    Test a single model
    
    Args:
        model_name: Name of model to test
        device: Device to run model on
        verbose: Print detailed information
    
    Returns:
        Dictionary with test results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name.upper()}")
        print(f"{'='*70}")
    
    try:
        # Create model
        model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
        model = model.to(device)
        
        # Print summary
        if verbose:
            model.print_summary()
        
        # Test forward pass with dummy data
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        # Verify output shape
        expected_shape = (batch_size, NUM_CLASSES)
        actual_shape = tuple(output.shape)
        
        if verbose:
            print(f"Input shape:  {tuple(dummy_input.shape)}")
            print(f"Output shape: {actual_shape}")
            print(f"Expected:     {expected_shape}")
        
        assert actual_shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {actual_shape}"
        
        # Test freeze/unfreeze
        if verbose:
            print(f"\nTesting freeze/unfreeze functionality:")
        
        # Freeze backbone
        model.freeze_backbone()
        trainable_frozen, total_frozen = model.get_trainable_params()
        
        if verbose:
            print(f"  After freeze: {trainable_frozen:,} / {total_frozen:,} trainable ({trainable_frozen/total_frozen*100:.1f}%)")
        
        # Unfreeze all
        model.unfreeze_backbone()
        trainable_unfrozen, total_unfrozen = model.get_trainable_params()
        
        if verbose:
            print(f"  After unfreeze: {trainable_unfrozen:,} / {total_unfrozen:,} trainable ({trainable_unfrozen/total_unfrozen*100:.1f}%)")
        
        # Get model info
        trainable, total = model.get_trainable_params()
        
        results = {
            'name': model_name,
            'success': True,
            'total_params': total,
            'trainable_params': trainable,
            'output_shape': actual_shape,
            'error': None
        }
        
        if verbose:
            print(f"\n✓ {model_name.upper()} - All tests passed!")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"\n✗ {model_name.upper()} - Test failed!")
            print(f"Error: {str(e)}")
        
        return {
            'name': model_name,
            'success': False,
            'total_params': 0,
            'trainable_params': 0,
            'output_shape': None,
            'error': str(e)
        }


def compare_models(results):
    """
    Print comparison table of all models
    
    Args:
        results: List of test result dictionaries
    """
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Total Params':<15} {'Status':<10} {'Notes'}")
    print(f"{'-'*70}")
    
    for result in results:
        name = result['name']
        params = f"{result['total_params']:,}" if result['success'] else "N/A"
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        notes = "" if result['success'] else result['error'][:30]
        
        print(f"{name:<20} {params:<15} {status:<10} {notes}")
    
    print(f"{'='*70}")
    
    # Summary
    total_tested = len(results)
    total_passed = sum(1 for r in results if r['success'])
    
    print(f"\nSummary: {total_passed}/{total_tested} models passed all tests")
    
    if total_passed == total_tested:
        print("✓ All models are ready for Phase 3 training!")
    else:
        print("⚠ Some models failed. Please review errors above.")


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare emotion recognition models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Test specific model (default: test all)')
    parser.add_argument('--all', action='store_true',
                       help='Test all available models')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Get device
    device_info = get_device_info()
    device = torch.device(device_info['device'])
    
    print("="*70)
    print("MODEL TESTING - PHASE 2 VALIDATION")
    print("="*70)
    print(f"Device:       {device_info['name']}")
    print(f"Input size:   {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Num classes:  {NUM_CLASSES}")
    print("="*70)
    
    # Determine which models to test
    if args.model:
        models_to_test = [args.model.lower()]
    else:
        models_to_test = ['mobilenet', 'resnet18', 'resnet50', 'efficientnet']
    
    # Test models
    results = []
    for model_name in models_to_test:
        result = test_model(model_name, device, verbose=not args.quiet)
        results.append(result)
    
    # Print comparison if testing multiple models
    if len(results) > 1:
        compare_models(results)
    
    # Exit code
    all_passed = all(r['success'] for r in results)
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
