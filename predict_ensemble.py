"""Phase 4: Ensemble Emotion Prediction System

Advanced ensemble system combining predictions from multiple models for improved accuracy.

Features:
- Loads multiple trained models (ResNet50, ResNet18, EfficientNet, MobileNet)
- Weighted or equal ensemble averaging
- Face detection preprocessing
- Detailed breakdown of individual model predictions
- Visualization of ensemble results
- Batch processing for folders

Usage:
    # Single image prediction
    python predict_ensemble.py --image test.jpg
    
    # Process entire folder
    python predict_ensemble.py --folder test_images/ --output_dir results/predictions
    
    # Custom model selection
    python predict_ensemble.py --image test.jpg --models resnet50,resnet18
    
    # Weighted ensemble (based on validation accuracy)
    python predict_ensemble.py --folder test_images/ --weights 0.4,0.3,0.15,0.15
    
    # With face detection
    python predict_ensemble.py --folder test_images/ --detect_face --visualize
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
import csv
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import EMOTION_LABELS, USE_GPU, GPU_ID
from models import create_model
from utils.data_loader import get_transforms


# Available models for ensemble
AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']

# Recommended weights based on Phase 3 validation accuracy
RECOMMENDED_WEIGHTS = {
    'resnet50': 0.40,    # 72.19% val acc
    'resnet18': 0.30,    # ~65% val acc
    'efficientnet': 0.15, # ~50% val acc
    'mobilenet': 0.15     # ~45% val acc
}


def get_device():
    """Get the appropriate device (GPU/CPU)."""
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def detect_face(image_path: Path) -> np.ndarray | None:
    """Detect and crop face from image using OpenCV Haar Cascade."""
    try:
        # Load cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARNING] Could not read image: {image_path}")
            return None
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("[WARNING] No face detected in image. Using full image.")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding (10%)
        padding = int(0.1 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        face_img = img[y1:y2, x1:x2]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        print(f"✓ Face detected at ({x}, {y}) with size {w}x{h}")
        return face_img
    
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return None


def load_ensemble_models(model_names: List[str], checkpoint_dir: Path, device) -> Dict:
    """Load multiple models for ensemble prediction.
    
    Args:
        model_names: List of model names to load
        checkpoint_dir: Directory containing model checkpoints
        device: Device to load models on
    
    Returns:
        Dictionary mapping model names to loaded models and their metadata
    """
    ensemble = {}
    
    print(f"\n{'='*70}")
    print("Loading Ensemble Models")
    print(f"{'='*70}")
    
    for model_name in model_names:
        try:
            # Construct checkpoint path
            checkpoint_path = checkpoint_dir / f'{model_name}_finetune_deep_best.pt'
            
            if not checkpoint_path.exists():
                print(f"[WARNING] Checkpoint not found for {model_name}: {checkpoint_path}")
                continue
            
            print(f"Loading {model_name}...")
            
            # Create model
            model = create_model(model_name, num_classes=7, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            model = model.to(device)
            model.eval()
            
            # Store model and metadata
            ensemble[model_name] = {
                'model': model,
                'val_acc': checkpoint.get('val_acc', 0.0),
                'checkpoint_path': checkpoint_path
            }
            
            print(f"  ✓ Loaded (Val Acc: {checkpoint.get('val_acc', 0)*100:.2f}%)")
        
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            continue
    
    if not ensemble:
        raise RuntimeError("Failed to load any models for ensemble!")
    
    print(f"{'='*70}")
    print(f"Successfully loaded {len(ensemble)} models\n")
    
    return ensemble


def preprocess_image(image_path: Path, detect_face_flag: bool = False) -> Tuple[torch.Tensor, str]:
    """Load and preprocess image for model input.
    
    Returns:
        Tuple of (preprocessed tensor, status message)
    """
    status = "Full image (no face detection)"
    
    # Detect face if requested
    if detect_face_flag:
        face_img = detect_face(image_path)
        if face_img is not None:
            img = Image.fromarray(face_img)
            status = "Face detected and cropped"
        else:
            img = Image.open(image_path).convert('RGB')
            status = "No face detected, using full image"
    else:
        img = Image.open(image_path).convert('RGB')
    
    # Get validation transforms (no augmentation)
    transforms = get_transforms('val')
    
    # Apply transforms
    img_tensor = transforms(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, status


def run_ensemble_inference(ensemble: Dict, img_tensor: torch.Tensor, device) -> Dict:
    """Run inference with all models in ensemble.
    
    Args:
        ensemble: Dictionary of loaded models
        img_tensor: Preprocessed image tensor
        device: Device for inference
    
    Returns:
        Dictionary mapping model names to prediction results
    """
    img_tensor = img_tensor.to(device)
    results = {}
    
    with torch.no_grad():
        for model_name, model_info in ensemble.items():
            model = model_info['model']
            
            # Get logits
            outputs = model(img_tensor)
            
            # Convert to probabilities (softmax)
            probs = torch.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, pred_idx = torch.max(probs, dim=1)
            
            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(probs, k=min(3, len(EMOTION_LABELS)), dim=1)
            
            results[model_name] = {
                'probabilities': probs[0].cpu().numpy(),
                'prediction': EMOTION_LABELS[pred_idx.item()],
                'confidence': confidence.item(),
                'top3': [(EMOTION_LABELS[idx.item()], prob.item()) 
                        for prob, idx in zip(top3_probs[0], top3_indices[0])]
            }
    
    return results


def aggregate_predictions(results: Dict, weights: Dict[str, float]) -> Dict:
    """Aggregate predictions using weighted averaging.
    
    Args:
        results: Dictionary of individual model predictions
        weights: Dictionary mapping model names to weights
    
    Returns:
        Dictionary with ensemble prediction results
    """
    # Collect probability vectors
    prob_arrays = []
    weight_values = []
    
    for model_name, result in results.items():
        prob_arrays.append(result['probabilities'])
        weight_values.append(weights.get(model_name, 1.0))
    
    # Stack probabilities
    all_probs = np.stack(prob_arrays)  # Shape: (num_models, num_classes)
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
    top3 = [(EMOTION_LABELS[idx], ensemble_probs[idx]) for idx in top3_indices]
    
    return {
        'probabilities': ensemble_probs,
        'prediction': EMOTION_LABELS[pred_idx],
        'confidence': confidence,
        'top3': top3,
        'weights_used': {name: weight_values[i] for i, name in enumerate(results.keys())}
    }


def print_ensemble_results(image_path: Path, preprocessing_status: str, 
                           individual_results: Dict, ensemble_result: Dict,
                           weights: Dict[str, float]):
    """Print formatted ensemble prediction results."""
    print(f"\n{'='*70}")
    print("ENSEMBLE EMOTION PREDICTION")
    print(f"{'='*70}")
    print(f"Image: {image_path.name}")
    print(f"Preprocessing: {preprocessing_status}")
    
    # Print weights
    weight_str = ", ".join([f"{name}: {w:.2f}" for name, w in ensemble_result['weights_used'].items()])
    print(f"Ensemble Weights: {weight_str}")
    
    # Individual predictions
    print(f"\nIndividual Model Predictions:")
    print(f"{'-'*70}")
    print(f"{'Model':<15} {'Prediction':<12} {'Confidence':<12} {'Top-3'}")
    print(f"{'-'*70}")
    
    for model_name, result in individual_results.items():
        top3_str = ", ".join([em for em, _ in result['top3']])
        print(f"{model_name.upper():<15} {result['prediction']:<12} {result['confidence']*100:>6.2f}%      {top3_str}")
    
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
    """Visualize ensemble predictions."""
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
    
    plt.suptitle(f'Ensemble Prediction: {ensemble_result["prediction"].upper()} '
                 f'({ensemble_result["confidence"]*100:.1f}%)', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        # Save to default location instead of showing (non-interactive environment)
        default_path = Path('ensemble_prediction.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {default_path}")
    
    plt.close()


def parse_weights(weights_str: str, model_names: List[str]) -> Dict[str, float]:
    """Parse weight string into dictionary.
    
    Args:
        weights_str: Comma-separated weights or 'equal' or 'recommended'
        model_names: List of model names
    
    Returns:
        Dictionary mapping model names to weights
    """
    if weights_str.lower() == 'equal':
        weight = 1.0 / len(model_names)
        return {name: weight for name in model_names}
    
    if weights_str.lower() == 'recommended':
        # Use recommended weights based on validation accuracy
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
        
        # Create dictionary
        weights = dict(zip(model_names, weight_values))
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    except Exception as e:
        raise ValueError(f"Invalid weights format: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 4: Ensemble Emotion Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                            help='Path to single input image')
    input_group.add_argument('--folder', type=str,
                            help='Path to folder containing images to process')
    
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated model names or "all" (default: all)')
    parser.add_argument('--weights', type=str, default='recommended',
                       help='Model weights: "equal", "recommended", or comma-separated values (default: recommended)')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--detect_face', action='store_true',
                       help='Enable face detection and cropping')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for predictions')
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Save visualization to specified path (single image mode only)')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--output_dir', type=str, default='ensemble_results',
                       help='Output directory for batch processing results (default: ensemble_results)')
    
    return parser.parse_args()


def process_single_image(image_path: Path, ensemble: Dict, device, weights: Dict,
                        detect_face_flag: bool, visualize: bool, save_viz_path: Path = None) -> Dict:
    """Process a single image and return results.
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_tensor, preprocessing_status = preprocess_image(image_path, detect_face_flag)
    
    # Run inference
    individual_results = run_ensemble_inference(ensemble, img_tensor, device)
    
    # Aggregate predictions
    ensemble_result = aggregate_predictions(individual_results, weights)
    
    # Add metadata
    result = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'preprocessing_status': preprocessing_status,
        'individual_predictions': individual_results,
        'ensemble_prediction': ensemble_result
    }
    
    # Visualize if requested
    if visualize or save_viz_path:
        visualize_ensemble(image_path, individual_results, ensemble_result, save_viz_path)
    
    return result


def process_folder(folder_path: Path, ensemble: Dict, device, weights: Dict,
                   detect_face_flag: bool, visualize: bool, output_dir: Path) -> List[Dict]:
    """Process all images in a folder.
    
    Returns:
        List of prediction results for all images
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"[WARNING] No images found in {folder_path}")
        return []
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"{'='*70}\n")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
    
    # Process each image
    all_results = []
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images"), 1):
        try:
            # Determine visualization path
            save_viz_path = None
            if visualize:
                save_viz_path = viz_dir / f'{image_path.stem}_ensemble.png'
            
            # Process image
            result = process_single_image(
                image_path, ensemble, device, weights,
                detect_face_flag, visualize, save_viz_path
            )
            
            # Print detailed results for this image
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


def save_results_to_json(results: List[Dict], output_path: Path):
    """Save batch prediction results to JSON."""
    # Convert numpy arrays to lists for JSON serialization
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
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
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
    for emotion in EMOTION_LABELS:
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


def main():
    args = parse_args()
    
    # Parse model names
    if args.models.lower() == 'all':
        model_names = AVAILABLE_MODELS
    else:
        model_names = [m.strip() for m in args.models.split(',') if m.strip() in AVAILABLE_MODELS]
    
    if not model_names:
        print("[ERROR] No valid models specified")
        return 1
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load ensemble models
    checkpoint_dir = Path(args.checkpoint_dir)
    ensemble = load_ensemble_models(model_names, checkpoint_dir, device)
    
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
        result = process_single_image(
            image_path, ensemble, device, weights,
            args.detect_face, args.visualize,
            Path(args.save_viz) if args.save_viz else None
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
        
        # Process all images
        results = process_folder(
            folder_path, ensemble, device, weights,
            args.detect_face, args.visualize, output_dir
        )
        
        if not results:
            print("[ERROR] No images were successfully processed")
            return 1
        
        # Save results
        if args.save_csv:
            csv_path = output_dir / 'predictions.csv'
            save_results_to_csv(results, csv_path)
        
        # Always save JSON
        json_path = output_dir / 'predictions.json'
        save_results_to_json(results, json_path)
        
        # Print summary
        print_batch_summary(results)
        
        print(f"All results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
