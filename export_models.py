"""Phase 5: Model Export for Deployment

Export trained PyTorch models to production-ready formats:
- ONNX: Cross-platform deployment (TensorFlow, CoreML, etc.)
- TorchScript: Optimized PyTorch format for C++ deployment

Features:
- Exports all 4 models (ResNet50, ResNet18, EfficientNet, MobileNet)
- Validates exported models match PyTorch accuracy
- Generates metadata files with preprocessing info
- Benchmarks inference speed comparison
- Supports dynamic batch sizes

Usage:
    python export_models.py --models all
    python export_models.py --models resnet50,resnet18 --format onnx
    python export_models.py --models mobilenet --validate --benchmark
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
import torch
import numpy as np
from tqdm import tqdm

from config import EMOTION_LABELS, USE_GPU, GPU_ID
from models import create_model


AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']
EXPORT_FORMATS = ['onnx', 'torchscript', 'both']


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def load_trained_model(model_name: str, checkpoint_dir: Path, device):
    """Load trained model from checkpoint."""
    checkpoint_path = checkpoint_dir / f'{model_name}_finetune_deep_best.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading {model_name} from {checkpoint_path}")
    
    # Create model
    model = create_model(model_name, num_classes=7, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint


def export_to_onnx(model, model_name: str, output_dir: Path, device):
    """Export model to ONNX format."""
    print(f"\n[ONNX] Exporting {model_name}...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Output path
    onnx_path = output_dir / 'onnx' / f'{model_name}.onnx'
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"[ONNX] Saved to: {onnx_path}")
    
    # Get file size
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[ONNX] File size: {file_size_mb:.2f} MB")
    
    return onnx_path


def export_to_torchscript(model, model_name: str, output_dir: Path, device):
    """Export model to TorchScript format."""
    print(f"\n[TorchScript] Exporting {model_name}...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Output path
    ts_path = output_dir / 'torchscript' / f'{model_name}_traced.pt'
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    traced_model.save(str(ts_path))
    
    print(f"[TorchScript] Saved to: {ts_path}")
    
    # Get file size
    file_size_mb = ts_path.stat().st_size / (1024 * 1024)
    print(f"[TorchScript] File size: {file_size_mb:.2f} MB")
    
    return ts_path


def validate_onnx_export(onnx_path: Path, pytorch_model, device):
    """Validate ONNX model produces same output as PyTorch."""
    try:
        import onnxruntime as ort
        
    except ImportError:
        print("[WARNING] onnxruntime not installed. Skipping ONNX validation.")
        print("          Install with: pip install onnxruntime")
        return False
    
    print(f"\n[VALIDATION] Checking ONNX model accuracy...")
    
    # Create ONNX session
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Test with random inputs
    num_tests = 10
    max_diff = 0.0
    
    for _ in range(num_tests):
        test_input = torch.randn(1, 3, 224, 224, device=device)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).cpu().numpy()
        
        # ONNX inference
        onnx_input = test_input.cpu().numpy()
        onnx_output = ort_session.run(None, {'input': onnx_input})[0]
        
        # Compare
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)
    
    print(f"[VALIDATION] Max difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("[VALIDATION] ✓ ONNX export validated successfully!")
        return True
    else:
        print("[VALIDATION] ⚠ ONNX output differs from PyTorch")
        return False


def validate_torchscript_export(ts_path: Path, pytorch_model, device):
    """Validate TorchScript model produces same output as PyTorch."""
    print(f"\n[VALIDATION] Checking TorchScript model accuracy...")
    
    # Load traced model
    traced_model = torch.jit.load(str(ts_path), map_location=device)
    traced_model.eval()
    
    # Test with random inputs
    num_tests = 10
    max_diff = 0.0
    
    for _ in range(num_tests):
        test_input = torch.randn(1, 3, 224, 224, device=device)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).cpu().numpy()
        
        # TorchScript inference
        with torch.no_grad():
            ts_output = traced_model(test_input).cpu().numpy()
        
        # Compare
        diff = np.abs(pytorch_output - ts_output).max()
        max_diff = max(max_diff, diff)
    
    print(f"[VALIDATION] Max difference: {max_diff:.6f}")
    
    if max_diff < 1e-7:
        print("[VALIDATION] ✓ TorchScript export validated successfully!")
        return True
    else:
        print("[VALIDATION] ⚠ TorchScript output differs from PyTorch")
        return False


def benchmark_inference(model_name: str, pytorch_model, onnx_path: Path, ts_path: Path, device):
    """Benchmark inference speed for different formats."""
    print(f"\n[BENCHMARK] Comparing inference speeds for {model_name}...")
    
    num_iterations = 100
    batch_size = 1
    
    # Warmup
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
    
    # Benchmark PyTorch
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pytorch_time = (time.time() - start) / num_iterations * 1000
    
    # Benchmark TorchScript
    traced_model = torch.jit.load(str(ts_path), map_location=device)
    traced_model.eval()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = traced_model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    ts_time = (time.time() - start) / num_iterations * 1000
    
    # Benchmark ONNX (if available)
    onnx_time = None
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        test_input = dummy_input.cpu().numpy()
        
        start = time.time()
        for _ in range(num_iterations):
            _ = ort_session.run(None, {'input': test_input})
        onnx_time = (time.time() - start) / num_iterations * 1000
    except ImportError:
        pass
    
    # Print results
    print(f"[BENCHMARK] Results (average over {num_iterations} iterations):")
    print(f"  PyTorch:     {pytorch_time:.3f} ms")
    print(f"  TorchScript: {ts_time:.3f} ms ({ts_time/pytorch_time:.2f}x)")
    if onnx_time:
        print(f"  ONNX:        {onnx_time:.3f} ms ({onnx_time/pytorch_time:.2f}x)")
    
    return {
        'pytorch_ms': pytorch_time,
        'torchscript_ms': ts_time,
        'onnx_ms': onnx_time
    }


def save_metadata(model_name: str, checkpoint, benchmark_results: dict, output_dir: Path):
    """Save model metadata and preprocessing info."""
    metadata = {
        'model_name': model_name,
        'architecture': model_name,
        'num_classes': 7,
        'class_labels': EMOTION_LABELS,
        'input_shape': [1, 3, 224, 224],
        'preprocessing': {
            'resize': [224, 224],
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'face_detection': 'MTCNN or Haar Cascade recommended'
        },
        'training_info': {
            'final_val_acc': checkpoint.get('val_acc', 0.0),
            'final_test_acc': checkpoint.get('test_acc', 0.0),
            'epoch': checkpoint.get('epoch', 0)
        },
        'benchmark': benchmark_results,
        'export_info': {
            'onnx_opset': 14,
            'dynamic_batch': True,
            'torch_version': torch.__version__
        }
    }
    
    # Save metadata
    metadata_path = output_dir / 'metadata' / f'{model_name}_metadata.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[METADATA] Saved to: {metadata_path}")


def export_model(
    model_name: str,
    checkpoint_dir: Path,
    output_dir: Path,
    export_format: str = 'both',
    validate: bool = True,
    benchmark: bool = False
):
    """Export a single model to specified format(s)."""
    print(f"\n{'='*70}")
    print(f"EXPORTING: {model_name.upper()}")
    print(f"{'='*70}")
    
    device = get_device()
    
    # Load model
    model, checkpoint = load_trained_model(model_name, checkpoint_dir, device)
    model = model.to(device)
    
    print(f"Model loaded - Val Acc: {checkpoint.get('val_acc', 0.0):.4f}")
    
    # Export to ONNX
    onnx_path = None
    if export_format in ['onnx', 'both']:
        onnx_path = export_to_onnx(model, model_name, output_dir, device)
        if validate:
            validate_onnx_export(onnx_path, model, device)
    
    # Export to TorchScript
    ts_path = None
    if export_format in ['torchscript', 'both']:
        ts_path = export_to_torchscript(model, model_name, output_dir, device)
        if validate:
            validate_torchscript_export(ts_path, model, device)
    
    # Benchmark
    benchmark_results = {}
    if benchmark and onnx_path and ts_path:
        benchmark_results = benchmark_inference(model_name, model, onnx_path, ts_path, device)
    
    # Save metadata
    save_metadata(model_name, checkpoint, benchmark_results, output_dir)
    
    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE: {model_name.upper()}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Export trained models for deployment')
    parser.add_argument('--models', type=str, default='all',
                       help=f'Models to export (comma-separated or "all"). Options: {", ".join(AVAILABLE_MODELS)}')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path('results/checkpoints'),
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=Path, default=Path('exported_models'),
                       help='Output directory for exported models')
    parser.add_argument('--format', type=str, default='both', choices=EXPORT_FORMATS,
                       help='Export format (onnx, torchscript, or both)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported models match PyTorch output')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark inference speed comparison')
    
    args = parser.parse_args()
    
    # Parse models
    if args.models.lower() == 'all':
        models = AVAILABLE_MODELS
    else:
        models = [m.strip() for m in args.models.split(',')]
        invalid = [m for m in models if m not in AVAILABLE_MODELS]
        if invalid:
            raise ValueError(f"Invalid models: {invalid}. Must be one of {AVAILABLE_MODELS}")
    
    print("="*70)
    print("PHASE 5: MODEL EXPORT FOR DEPLOYMENT")
    print("="*70)
    print(f"Models: {', '.join(models)}")
    print(f"Format: {args.format}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Validate: {args.validate}")
    print(f"Benchmark: {args.benchmark}")
    print("="*70)
    
    # Export each model
    for model_name in models:
        try:
            export_model(
                model_name,
                args.checkpoint_dir,
                args.output_dir,
                args.format,
                args.validate,
                args.benchmark
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to export {model_name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("ALL EXPORTS COMPLETE!")
    print("="*70)
    print(f"\nExported models saved to: {args.output_dir}")
    print("\nUsage examples:")
    print("  - ONNX: Load with onnxruntime (Python, C++, C#, Java)")
    print("  - TorchScript: Load with torch.jit.load() (Python, C++)")
    print("\nNext steps:")
    print("  1. Deploy ONNX models to mobile/edge devices")
    print("  2. Use TorchScript for C++ production servers")
    print("  3. Consider quantization for further optimization")


if __name__ == '__main__':
    main()
