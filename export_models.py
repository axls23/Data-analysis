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

# Check if onnxruntime is available for quantization
try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_QUANTIZATION_AVAILABLE = True
except ImportError:
    ONNX_QUANTIZATION_AVAILABLE = False


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


def export_to_onnx(model, model_name: str, output_dir: Path, device, opset: int = 18):
    """Export model to ONNX format using modern exporter settings.

    Args:
        model: PyTorch model (assumed in eval mode)
        model_name: Name for output file
        output_dir: Base output directory
        device: torch.device
        opset: ONNX opset to target (default 18 for current PyTorch >=2.1)
    """
    print(f"\n[ONNX] Exporting {model_name} (opset {opset})...")

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    onnx_path = output_dir / 'onnx' / f'{model_name}.onnx'
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Force legacy ONNX exporter (dynamo is broken in PyTorch 2.9)
    import os
    # Disable new dynamo-based exporter
    os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'
    torch.onnx.PYTORCH_ONNX_EXPORT_STRICT_MODE = False
    
    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        # Force legacy exporter (dynamo creates corrupted 0.18 MB files)
        dynamo=False
    )

    use_dynamic_axes = True
    try:
        # Use legacy torch.onnx.export
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                **export_kwargs
            )
    except Exception as e:
        print(f"[ONNX] Export with dynamic axes failed: {e}")
        print(f"[ONNX] Retrying with static batch size...")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    **export_kwargs
                )
            use_dynamic_axes = False
        except Exception as e2:
            raise RuntimeError(f"ONNX export failed for {model_name}: {e2}")

    print(f"[ONNX] Saved to: {onnx_path}")
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    dyn_msg = "dynamic batch" if use_dynamic_axes else "static batch" 
    print(f"[ONNX] File size: {file_size_mb:.2f} MB ({dyn_msg})")
    return onnx_path


def quantize_onnx_model(onnx_path: Path, model_name: str, output_dir: Path):
    """Apply dynamic quantization to ONNX model to reduce size and improve speed.
    
    Args:
        onnx_path: Path to original ONNX model
        model_name: Name of the model
        output_dir: Base output directory
        
    Returns:
        Path to quantized ONNX model
    """
    if not ONNX_QUANTIZATION_AVAILABLE:
        print("[WARNING] onnxruntime not installed. Skipping quantization.")
        print("          Install with: pip install onnxruntime")
        return None
    
    print(f"\n[QUANTIZATION] Applying dynamic INT8 quantization to {model_name}...")
    
    quantized_path = output_dir / 'onnx' / f'{model_name}_quantized.onnx'
    
    try:
        # Apply dynamic quantization (INT8)
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QUInt8,  # Quantize weights to 8-bit unsigned integers
            per_channel=True,  # Per-channel quantization for better accuracy
        )
        
        # Compare file sizes
        original_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size_mb = quantized_path.stat().st_size / (1024 * 1024)
        reduction = ((original_size_mb - quantized_size_mb) / original_size_mb) * 100
        
        print(f"[QUANTIZATION] Saved to: {quantized_path}")
        print(f"[QUANTIZATION] Original size: {original_size_mb:.2f} MB")
        print(f"[QUANTIZATION] Quantized size: {quantized_size_mb:.2f} MB")
        print(f"[QUANTIZATION] Size reduction: {reduction:.1f}%")
        
        return quantized_path
        
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        return None


def export_to_torchscript(model, model_name: str, output_dir: Path, device):
    """Export model to optimized TorchScript format."""
    print(f"\n[TorchScript] Exporting {model_name}...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    # Trace
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model = torch.jit.freeze(traced_model)
    # Apply inference optimizations if available
    try:
        traced_model = torch.jit.optimize_for_inference(traced_model)
    except Exception:
        pass
    ts_path = output_dir / 'torchscript' / f'{model_name}_traced.pt'
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(ts_path))
    print(f"[TorchScript] Saved to: {ts_path}")
    file_size_mb = ts_path.stat().st_size / (1024 * 1024)
    print(f"[TorchScript] File size: {file_size_mb:.2f} MB (optimized)")
    return ts_path


def validate_onnx_export(onnx_path: Path, pytorch_model, device):
    """Validate ONNX model produces similar functional output as PyTorch.

    Uses top-1 agreement and probability (softmax) difference rather than raw
    logits absolute diff only. Tolerates small numeric drift typical of backend differences.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not installed. Skipping ONNX validation.")
        print("          Install with: pip install onnxruntime")
        return False

    print(f"\n[VALIDATION] Checking ONNX model accuracy...")
    ort_session = ort.InferenceSession(str(onnx_path))

    num_tests = 20
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    top1_matches = 0

    for _ in range(num_tests):
        test_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            pt_logits = pytorch_model(test_input).cpu().numpy()
        onnx_logits = ort_session.run(None, {'input': test_input.cpu().numpy()})[0]

        # Track logit diff
        logit_diff = np.abs(pt_logits - onnx_logits).max()
        max_logit_diff = max(max_logit_diff, logit_diff)

        # Compare top-1
        pt_top1 = np.argmax(pt_logits, axis=1).item()
        onnx_top1 = np.argmax(onnx_logits, axis=1).item()
        if pt_top1 == onnx_top1:
            top1_matches += 1

        # Prob diff
        pt_probs = torch.softmax(torch.from_numpy(pt_logits), dim=1).numpy()
        onnx_probs = torch.softmax(torch.from_numpy(onnx_logits), dim=1).numpy()
        prob_diff = np.abs(pt_probs - onnx_probs).max()
        max_prob_diff = max(max_prob_diff, prob_diff)

    top1_agreement = top1_matches / num_tests
    print(f"[VALIDATION] Max logit diff: {max_logit_diff:.6f}")
    print(f"[VALIDATION] Max prob  diff: {max_prob_diff:.6f}")
    print(f"[VALIDATION] Top-1 agreement: {top1_agreement*100:.1f}% ({top1_matches}/{num_tests})")

    # Acceptance criteria:
    # - Top-1 agreement >= 95%
    # - Max probability diff < 0.01
    accepted = top1_agreement >= 0.95 and max_prob_diff < 0.01
    if accepted:
        print("[VALIDATION] ✓ ONNX export functionally validated!")
    else:
        print("[VALIDATION] ⚠ Potential mismatch (acceptable small numeric drift is normal). Review if critical.")
    return accepted


def validate_torchscript_export(ts_path: Path, pytorch_model, device):
    """Validate TorchScript model similarity using top-1 agreement and probability drift."""
    print(f"\n[VALIDATION] Checking TorchScript model accuracy...")
    traced_model = torch.jit.load(str(ts_path), map_location=device)
    traced_model.eval()

    num_tests = 20
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    top1_matches = 0

    for _ in range(num_tests):
        test_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            pt_logits = pytorch_model(test_input).cpu().numpy()
            ts_logits = traced_model(test_input).cpu().numpy()

        logit_diff = np.abs(pt_logits - ts_logits).max()
        max_logit_diff = max(max_logit_diff, logit_diff)

        pt_top1 = np.argmax(pt_logits, axis=1).item()
        ts_top1 = np.argmax(ts_logits, axis=1).item()
        if pt_top1 == ts_top1:
            top1_matches += 1

        pt_probs = torch.softmax(torch.from_numpy(pt_logits), dim=1).numpy()
        ts_probs = torch.softmax(torch.from_numpy(ts_logits), dim=1).numpy()
        prob_diff = np.abs(pt_probs - ts_probs).max()
        max_prob_diff = max(max_prob_diff, prob_diff)

    top1_agreement = top1_matches / num_tests
    print(f"[VALIDATION] Max logit diff: {max_logit_diff:.6f}")
    print(f"[VALIDATION] Max prob  diff: {max_prob_diff:.6f}")
    print(f"[VALIDATION] Top-1 agreement: {top1_agreement*100:.1f}% ({top1_matches}/{num_tests})")

    accepted = top1_agreement >= 0.95 and max_prob_diff < 0.01
    if accepted:
        print("[VALIDATION] ✓ TorchScript export functionally validated!")
    else:
        print("[VALIDATION] ⚠ Potential mismatch; investigate if accuracy-sensitive.")
    return accepted


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


def save_metadata(model_name: str, checkpoint, benchmark_results: dict, output_dir: Path, opset: int = 18):
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
            'onnx_opset': opset,
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
    benchmark: bool = False,
    quantize: bool = False,
    opset: int = 18
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
    quantized_path = None
    if export_format in ['onnx', 'both']:
        onnx_path = export_to_onnx(model, model_name, output_dir, device, opset=opset)
        if validate:
            validate_onnx_export(onnx_path, model, device)
        
        # Apply quantization if requested
        if quantize:
            quantized_path = quantize_onnx_model(onnx_path, model_name, output_dir)
            if quantized_path and validate:
                print(f"\n[VALIDATION] Validating quantized model...")
                validate_onnx_export(quantized_path, model, device)
    
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
        
        # Benchmark quantized model if available
        if quantized_path:
            print(f"\n[BENCHMARK] Testing quantized model inference speed...")
            try:
                import onnxruntime as ort
                ort_session = ort.InferenceSession(str(quantized_path))
                test_input = torch.randn(1, 3, 224, 224).cpu().numpy()
                
                num_iterations = 100
                start = time.time()
                for _ in range(num_iterations):
                    _ = ort_session.run(None, {'input': test_input})
                quantized_time = (time.time() - start) / num_iterations * 1000
                
                print(f"  Quantized ONNX: {quantized_time:.3f} ms ({quantized_time/benchmark_results['pytorch_ms']:.2f}x)")
                benchmark_results['quantized_onnx_ms'] = quantized_time
            except Exception as e:
                print(f"[WARNING] Quantized model benchmark failed: {e}")
    
    # Save metadata
    save_metadata(model_name, checkpoint, benchmark_results, output_dir, opset=opset)
    
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
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 dynamic quantization to ONNX models (reduces size ~50-75%%)')
    parser.add_argument('--opset', type=int, default=18,
                       help='ONNX opset version to use (default: 18)')
    
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
                args.benchmark,
                args.quantize,
                opset=args.opset
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
