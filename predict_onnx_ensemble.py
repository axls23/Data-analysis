"""ONNX Ensemble Emotion Prediction System

Lightweight ensemble system using only ONNX models - no PyTorch required!
Perfect for deployment and sharing with teammates.

Dependencies:
    pip install onnxruntime opencv-python pillow numpy

Features:
- Weighted ensemble of multiple ONNX models
- Automatic quantized model selection (validated models only)
- Face detection preprocessing
- Batch processing
- CSV export

Usage:
    # Single image prediction
    python predict_onnx_ensemble.py --image test.jpg
    
    # With face detection
    python predict_onnx_ensemble.py --image test.jpg --detect_face
    
    # Process entire folder
    python predict_onnx_ensemble.py --folder test_images/ --detect_face --quiet
    
    # Custom model selection
    python predict_onnx_ensemble.py --image test.jpg --models resnet50,resnet18
    
    # Custom weights
    python predict_onnx_ensemble.py --image test.jpg --weights 0.4,0.3,0.2,0.1
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
import csv
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed!")
    print("Install with: pip install onnxruntime")
    exit(1)


# Emotion labels (must match training order)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Available models
AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']

# Models that should use quantized versions (based on validation)
QUANTIZATION_WHITELIST = {
    'mobilenet': True,     # 100% agreement, 72.8% size reduction
    'efficientnet': False, # 0% agreement - broken after quantization
    'resnet18': True,      # 100% agreement, 74.9% size reduction
    'resnet50': True,      # 100% agreement, 74.8% size reduction
}

# Recommended weights based on Phase 3 validation accuracy
RECOMMENDED_WEIGHTS = {
    'resnet50': 0.40,    # 72.19% val acc
    'resnet18': 0.30,    # ~65% val acc
    'efficientnet': 0.15, # ~50% val acc
    'mobilenet': 0.15     # ~45% val acc
}


def detect_face(image_path: Path, verbose: bool = True) -> np.ndarray | None:
    """Detect and crop face from image using OpenCV Haar Cascade."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        img = cv2.imread(str(image_path))
        if img is None:
            if verbose:
                print(f"[WARNING] Could not read image: {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            if verbose:
                print("[WARNING] No face detected in image. Using full image.")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        padding = int(0.1 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        face_img = img[y1:y2, x1:x2]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        if verbose:
            print(f"✓ Face detected at ({x}, {y}) with size {w}x{h}")
        return face_img
    
    except Exception as e:
        if verbose:
            print(f"[ERROR] Face detection failed: {e}")
        return None


def preprocess_image(image_path: Path, detect_face_flag: bool = False, verbose: bool = True) -> Tuple[np.ndarray, str]:
    """Load and preprocess image for model input."""
    status = "Full image (no face detection)"
    
    if detect_face_flag:
        face_img = detect_face(image_path, verbose=verbose)
        if face_img is not None:
            img = Image.fromarray(face_img)
            status = "Face detected and cropped"
        else:
            img = Image.open(image_path).convert('RGB')
            status = "No face detected, using full image"
    else:
        img = Image.open(image_path).convert('RGB')
    
    # Resize to 224×224
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Convert to array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Transpose to (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Add batch dimension
    return img_array[np.newaxis, :].astype(np.float32), status


def load_onnx_models(model_names: List[str], onnx_dir: Path, verbose: bool = True) -> Dict:
    """Load multiple ONNX models for ensemble prediction."""
    ensemble = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print("Loading ONNX Ensemble Models")
        print(f"{'='*70}")
    
    for model_name in model_names:
        try:
            # Determine which model to load
            use_quantized = False
            if QUANTIZATION_WHITELIST.get(model_name, False):
                quantized_path = onnx_dir / f'{model_name}_quantized.onnx'
                if quantized_path.exists():
                    model_path = quantized_path
                    use_quantized = True
                else:
                    model_path = onnx_dir / f'{model_name}.onnx'
            else:
                model_path = onnx_dir / f'{model_name}.onnx'
            
            if not model_path.exists():
                if verbose:
                    print(f"[WARNING] Model not found for {model_name}: {model_path}")
                continue
            
            if verbose:
                print(f"Loading {model_name}{'_quantized' if use_quantized else ''}...")
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(model_path))
            
            # Load metadata if available
            metadata_path = Path('exported_models/metadata') / f'{model_name}_metadata.json'
            val_acc = 0.0
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    val_acc = metadata.get('training_info', {}).get('final_val_acc', 0.0)
            
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            ensemble[model_name] = {
                'session': session,
                'val_acc': val_acc,
                'model_path': model_path,
                'is_quantized': use_quantized,
                'size_mb': file_size_mb
            }
            
            if verbose:
                quant_str = " (quantized)" if use_quantized else ""
                print(f"  ✓ Loaded{quant_str} - Val Acc: {val_acc*100:.2f}%, Size: {file_size_mb:.2f} MB")
        
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load {model_name}: {e}")
            continue
    
    if not ensemble:
        raise RuntimeError("Failed to load any models for ensemble!")
    
    if verbose:
        print(f"{'='*70}")
        print(f"Successfully loaded {len(ensemble)} models\n")
    
    return ensemble


def run_ensemble_inference(ensemble: Dict, img_tensor: np.ndarray) -> Dict:
    """Run inference with all models in ensemble."""
    results = {}
    
    for model_name, model_info in ensemble.items():
        session = model_info['session']
        
        # Run inference
        outputs = session.run(None, {'input': img_tensor})
        logits = outputs[0][0]
        
        # Convert to probabilities (softmax)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Get top prediction
        pred_idx = np.argmax(probs)
        
        # Get top-3
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        results[model_name] = {
            'probabilities': probs,
            'prediction': EMOTIONS[pred_idx],
            'confidence': probs[pred_idx],
            'top3': [(EMOTIONS[idx], probs[idx]) for idx in top3_idx]
        }
    
    return results


def aggregate_predictions(results: Dict, weights: Dict[str, float]) -> Dict:
    """Aggregate predictions using weighted averaging."""
    # Collect probability vectors
    prob_arrays = []
    weight_values = []
    
    for model_name, result in results.items():
        prob_arrays.append(result['probabilities'])
        weight_values.append(weights.get(model_name, 1.0))
    
    # Stack probabilities
    all_probs = np.stack(prob_arrays)
    weight_values = np.array(weight_values)
    
    # Normalize weights
    weight_values = weight_values / weight_values.sum()
    
    # Weighted average
    ensemble_probs = np.average(all_probs, axis=0, weights=weight_values)
    
    # Get top prediction
    pred_idx = np.argmax(ensemble_probs)
    confidence = ensemble_probs[pred_idx]
    
    # Get top-3
    top3_indices = np.argsort(ensemble_probs)[-3:][::-1]
    top3 = [(EMOTIONS[idx], ensemble_probs[idx]) for idx in top3_indices]
    
    return {
        'probabilities': ensemble_probs,
        'prediction': EMOTIONS[pred_idx],
        'confidence': confidence,
        'top3': top3,
        'weights_used': {name: weight_values[i] for i, name in enumerate(results.keys())}
    }


def print_ensemble_results(image_path: Path, preprocessing_status: str,
                           individual_results: Dict, ensemble_result: Dict,
                           weights: Dict[str, float]):
    """Print formatted ensemble prediction results."""
    print(f"\n{'='*70}")
    print("ONNX ENSEMBLE EMOTION PREDICTION")
    print(f"{'='*70}")
    print(f"Image: {image_path.name}")
    print(f"Preprocessing: {preprocessing_status}")
    
    # Print weights
    weight_str = ", ".join([f"{name}: {w:.2f}" for name, w in ensemble_result['weights_used'].items()])
    print(f"Ensemble Weights: {weight_str}")
    
    # Individual predictions
    print(f"\nIndividual Model Predictions:")
    print(f"{'-'*70}")
    print(f"{'Model':<20} {'Prediction':<12} {'Confidence':<12} {'Type'}")
    print(f"{'-'*70}")
    
    for model_name, result in individual_results.items():
        # Check if quantized
        model_type = "quantized" if model_name in ['mobilenet', 'resnet18', 'resnet50'] else "original"
        print(f"{model_name.upper():<20} {result['prediction']:<12} {result['confidence']*100:>6.2f}%      {model_type}")
    
    print(f"{'-'*70}")
    
    # Ensemble result
    print(f"\nEnsemble Result (Weighted Average):")
    print(f"{'-'*70}")
    print(f"PREDICTED EMOTION: {ensemble_result['prediction'].upper()}")
    print(f"CONFIDENCE: {ensemble_result['confidence']*100:.2f}%")
    print(f"{'-'*70}")
    
    # Top-3 ensemble predictions with bars
    print(f"\nTop-3 Ensemble Predictions:")
    for i, (emotion, conf) in enumerate(ensemble_result['top3'], 1):
        bar_length = int(conf * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{i}. {emotion.upper():<12} {conf*100:>6.2f}%  {bar}")
    
    print(f"{'='*70}\n")


def visualize_ensemble(image_path: Path, individual_results: Dict,
                      ensemble_result: Dict, save_path: Path = None):
    """Visualize ensemble predictions with matplotlib."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Show image
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('Input Image', fontsize=12, fontweight='bold')
    
    # Individual model predictions
    ax_individual = fig.add_subplot(gs[0, 1:])
    models = list(individual_results.keys())
    confidences = [individual_results[m]['confidence'] * 100 for m in models]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax_individual.barh(range(len(models)), confidences, color=colors[:len(models)])
    ax_individual.set_yticks(range(len(models)))
    ax_individual.set_yticklabels([m.upper() for m in models])
    ax_individual.set_xlabel('Confidence (%)', fontweight='bold')
    ax_individual.set_title('Individual Model Predictions', fontsize=12, fontweight='bold')
    ax_individual.set_xlim([0, 105])
    
    # Add labels
    for i, (bar, conf, model) in enumerate(zip(bars, confidences, models)):
        pred = individual_results[model]['prediction']
        ax_individual.text(conf + 2, i, f'{pred.upper()} ({conf:.1f}%)',
                          va='center', fontweight='bold', fontsize=9)
    
    # Ensemble top-3
    ax_ensemble = fig.add_subplot(gs[1, 1:])
    emotions = [e for e, _ in ensemble_result['top3']]
    ensemble_confs = [c * 100 for _, c in ensemble_result['top3']]
    
    bars = ax_ensemble.barh(emotions, ensemble_confs, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax_ensemble.set_xlabel('Confidence (%)', fontweight='bold')
    ax_ensemble.set_title('Ensemble Top-3 Predictions', fontsize=12, fontweight='bold')
    ax_ensemble.set_xlim([0, 105])
    
    # Add percentage labels
    for bar, conf in zip(bars, ensemble_confs):
        ax_ensemble.text(conf + 2, bar.get_y() + bar.get_height()/2,
                        f'{conf:.1f}%', va='center', fontweight='bold')
    
    plt.suptitle(f'ONNX Ensemble Prediction: {ensemble_result["prediction"].upper()} '
                 f'({ensemble_result["confidence"]*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    else:
        # Save to default location
        default_path = Path('onnx_ensemble_prediction.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {default_path}")
    
    plt.close()


def process_single_image(image_path: Path, ensemble: Dict, weights: Dict,
                        detect_face_flag: bool, visualize: bool = False,
                        save_viz_path: Path = None, verbose: bool = True) -> Dict:
    """Process a single image and return results."""
    # Preprocess image
    img_tensor, preprocessing_status = preprocess_image(image_path, detect_face_flag, verbose=verbose)
    
    # Run inference
    individual_results = run_ensemble_inference(ensemble, img_tensor)
    
    # Aggregate predictions
    ensemble_result = aggregate_predictions(individual_results, weights)
    
    # Generate visualization if requested
    if visualize:
        visualize_ensemble(image_path, individual_results, ensemble_result, save_viz_path)
    
    # Add metadata
    result = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'preprocessing_status': preprocessing_status,
        'individual_predictions': individual_results,
        'ensemble_prediction': ensemble_result
    }
    
    return result


def process_folder(folder_path: Path, ensemble: Dict, weights: Dict,
                   detect_face_flag: bool, visualize: bool = False,
                   output_dir: Path = None, verbose: bool = True) -> List[Dict]:
    """Process all images in a folder."""
    # Create visualization directory if needed
    viz_dir = None
    if visualize and output_dir:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = set()
    
    for ext in image_extensions:
        image_files.update(folder_path.glob(f'*{ext}'))
        image_files.update(folder_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(image_files))
    
    if not image_files:
        print(f"[WARNING] No images found in {folder_path}")
        return []
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"{'='*70}\n")
    
    # Process each image
    all_results = []
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images"), 1):
        try:
            # Determine visualization save path
            viz_path = None
            if visualize and viz_dir:
                viz_path = viz_dir / f"{image_path.stem}_prediction.png"
            
            # Process image
            result = process_single_image(
                image_path, ensemble, weights,
                detect_face_flag, visualize=visualize,
                save_viz_path=viz_path, verbose=verbose
            )
            
            # Print detailed results for this image only in verbose mode
            if verbose:
                print(f"\n{'='*70}")
                print(f"Image {idx}/{len(image_files)}: {image_path.name}")
                print(f"{'='*70}")
                print_ensemble_results(image_path,
                                      result['preprocessing_status'],
                                      result['individual_predictions'],
                                      result['ensemble_prediction'],
                                      weights)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {image_path.name}: {e}")
            continue
    
    return all_results


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save batch prediction results to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Image', 'Ensemble_Prediction', 'Ensemble_Confidence']
        
        # Add individual model columns
        if results:
            first_result = results[0]
            for model_name in first_result['individual_predictions'].keys():
                header.extend([f'{model_name.upper()}_Prediction', f'{model_name.upper()}_Confidence'])
        
        header.extend(['Preprocessing_Status', 'Top3_Predictions'])
        writer.writerow(header)
        
        # Data rows
        for result in results:
            row = [
                result['image_name'],
                result['ensemble_prediction']['prediction'],
                f"{result['ensemble_prediction']['confidence']*100:.2f}%"
            ]
            
            # Individual predictions
            for model_name, pred_data in result['individual_predictions'].items():
                row.extend([
                    pred_data['prediction'],
                    f"{pred_data['confidence']*100:.2f}%"
                ])
            
            # Additional info
            row.append(result['preprocessing_status'])
            
            # Top-3 ensemble predictions
            top3_str = " | ".join([f"{em}: {conf*100:.1f}%"
                                  for em, conf in result['ensemble_prediction']['top3']])
            row.append(top3_str)
            
            writer.writerow(row)
    
    print(f"Results saved to: {output_path}")


def print_batch_summary(results: List[Dict]):
    """Print summary statistics for batch processing."""
    if not results:
        return
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(results)}")
    
    # Count ensemble predictions
    ensemble_counts = {}
    for result in results:
        pred = result['ensemble_prediction']['prediction']
        ensemble_counts[pred] = ensemble_counts.get(pred, 0) + 1
    
    print(f"\nEnsemble Prediction Distribution:")
    print(f"{'-'*70}")
    for emotion in EMOTIONS:
        count = ensemble_counts.get(emotion, 0)
        percentage = (count / len(results)) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{emotion.upper():<12} {count:>3} ({percentage:>5.1f}%)  {bar}")
    
    # Average confidence
    avg_confidence = np.mean([r['ensemble_prediction']['confidence'] for r in results])
    print(f"\nAverage Ensemble Confidence: {avg_confidence*100:.2f}%")
    
    # Face detection stats
    face_detected = sum(1 for r in results if 'detected' in r['preprocessing_status'].lower())
    print(f"Images with face detected: {face_detected}/{len(results)} ({face_detected/len(results)*100:.1f}%)")
    
    print(f"{'='*70}\n")


def parse_weights(weights_str: str, model_names: List[str]) -> Dict[str, float]:
    """Parse weight string into dictionary."""
    if weights_str.lower() == 'equal':
        weight = 1.0 / len(model_names)
        return {name: weight for name in model_names}
    
    if weights_str.lower() == 'recommended':
        weights = {}
        for name in model_names:
            weights[name] = RECOMMENDED_WEIGHTS.get(name, 0.25)
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    # Parse comma-separated values
    try:
        weight_values = [float(w.strip()) for w in weights_str.split(',')]
        
        if len(weight_values) != len(model_names):
            raise ValueError(f"Number of weights ({len(weight_values)}) must match "
                           f"number of models ({len(model_names)})")
        
        weights = dict(zip(model_names, weight_values))
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    except Exception as e:
        raise ValueError(f"Invalid weights format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='ONNX Ensemble Emotion Prediction System (No PyTorch Required)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                            help='Path to single input image')
    input_group.add_argument('--folder', type=str,
                            help='Path to folder containing images to process')
    
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated model names or "all" (default: all)')
    parser.add_argument('--weights', type=str, default='recommended',
                       help='Model weights: "equal", "recommended", or comma-separated values (default: recommended)')
    parser.add_argument('--onnx_dir', type=str, default='exported_models/onnx',
                       help='Directory containing ONNX models')
    parser.add_argument('--detect_face', action='store_true',
                       help='Enable face detection and cropping')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for predictions')
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Save visualization to specified path (single image mode only)')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--output_dir', type=str, default='onnx_ensemble_results',
                       help='Output directory for batch processing results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output during batch processing')
    
    args = parser.parse_args()
    
    # Parse model names
    if args.models.lower() == 'all':
        model_names = AVAILABLE_MODELS
    else:
        model_names = [m.strip() for m in args.models.split(',') if m.strip() in AVAILABLE_MODELS]
    
    if not model_names:
        print("[ERROR] No valid models specified")
        return 1
    
    print(f"Using ONNX models (no PyTorch required)")
    
    # Load ensemble models
    onnx_dir = Path(args.onnx_dir)
    ensemble = load_onnx_models(model_names, onnx_dir, verbose=not args.quiet)
    
    # Update model_names to only loaded models
    model_names = list(ensemble.keys())
    
    # Parse weights
    try:
        weights = parse_weights(args.weights, model_names)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    
    # Process images
    if args.image:
        # Single image mode
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            return 1
        
        print("Preprocessing image...")
        
        # Determine visualization path
        viz_path = None
        if args.visualize:
            if args.save_viz:
                viz_path = Path(args.save_viz)
            else:
                viz_path = None  # Will use default path
        
        result = process_single_image(
            image_path, ensemble, weights,
            args.detect_face, visualize=args.visualize,
            save_viz_path=viz_path, verbose=True
        )
        
        # Display results
        print_ensemble_results(
            image_path,
            result['preprocessing_status'],
            result['individual_predictions'],
            result['ensemble_prediction'],
            weights
        )
        
    else:
        # Folder mode
        folder_path = Path(args.folder)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"[ERROR] Folder not found: {folder_path}")
            return 1
        
        output_dir = Path(args.output_dir)
        verbose = not args.quiet
        
        # Process all images
        results = process_folder(
            folder_path, ensemble, weights,
            args.detect_face, visualize=args.visualize,
            output_dir=output_dir, verbose=verbose
        )
        
        if not results:
            print("[ERROR] No images were successfully processed")
            return 1
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_csv:
            csv_path = output_dir / 'predictions.csv'
            save_results_to_csv(results, csv_path)
        
        # Always save JSON
        json_path = output_dir / 'predictions.json'
        json_results = []
        for result in results:
            json_result = {
                'image_name': result['image_name'],
                'image_path': result['image_path'],
                'preprocessing_status': result['preprocessing_status'],
                'ensemble_prediction': {
                    'prediction': result['ensemble_prediction']['prediction'],
                    'confidence': float(result['ensemble_prediction']['confidence']),
                    'top3': [(em, float(conf)) for em, conf in result['ensemble_prediction']['top3']],
                    'weights_used': result['ensemble_prediction']['weights_used']
                },
                'individual_predictions': {}
            }
            
            for model_name, pred_data in result['individual_predictions'].items():
                json_result['individual_predictions'][model_name] = {
                    'prediction': pred_data['prediction'],
                    'confidence': float(pred_data['confidence']),
                    'top3': [(em, float(conf)) for em, conf in pred_data['top3']]
                }
            
            json_results.append(json_result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {json_path}")
        
        # Print summary
        print_batch_summary(results)
        
        print(f"All results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
