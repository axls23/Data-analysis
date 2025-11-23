"""Phase 4: Single Image Emotion Prediction CLI

Interactive command-line tool for emotion detection on single images.

Features:
- Load any trained model (MobileNet, EfficientNet, ResNet18, ResNet50)
- Predict emotion from image file
- Display top-3 predictions with confidence scores
- Preprocessing with face detection option
- Visualization of prediction results

Usage:
    # Basic prediction
    python predict.py --image path/to/image.jpg --model resnet50
    
    # With face detection
    python predict.py --image path/to/image.jpg --model resnet50 --detect_face
    
    # Show visualization
    python predict.py --image path/to/image.jpg --model resnet18 --visualize
    
    # Specify checkpoint
    python predict.py --image test.jpg --model mobilenet --checkpoint results/checkpoints/mobilenet_finetune_deep_best.pt
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import EMOTION_LABELS, USE_GPU, GPU_ID, IMAGE_SIZE
from models import create_model
from utils.data_loader import get_transforms


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def load_model(model_name: str, checkpoint_path: Path, device):
    """Load trained model from checkpoint."""
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    # Create model
    model = create_model(model_name, num_classes=7, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'val_acc' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    return model


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
        
        print(f"Face detected at ({x}, {y}) with size {w}x{h}")
        return face_img
    
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return None


def preprocess_image(image_path: Path, detect_face_flag: bool = False) -> torch.Tensor:
    """Load and preprocess image for model input."""
    # Detect face if requested
    if detect_face_flag:
        face_img = detect_face(image_path)
        if face_img is not None:
            img = Image.fromarray(face_img)
        else:
            img = Image.open(image_path).convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')
    
    # Get validation transforms (no augmentation)
    transforms = get_transforms(is_training=False)
    
    # Apply transforms
    img_tensor = transforms(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def predict_emotion(model, img_tensor: torch.Tensor, device, top_k: int = 3):
    """Run inference and get top-k predictions."""
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(EMOTION_LABELS)), dim=1)
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            'emotion': EMOTION_LABELS[idx.item()],
            'confidence': prob.item()
        })
    
    return results


def visualize_prediction(image_path: Path, predictions: list, save_path: Path = None):
    """Visualize image with prediction results."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=12, fontweight='bold')
    
    # Show predictions
    emotions = [p['emotion'] for p in predictions]
    confidences = [p['confidence'] * 100 for p in predictions]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax2.barh(emotions, confidences, color=colors[:len(emotions)])
    ax2.set_xlabel('Confidence (%)', fontweight='bold')
    ax2.set_title('Top Predictions', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 105])
    
    # Add percentage labels
    for bar, conf in zip(bars, confidences):
        ax2.text(conf + 2, bar.get_y() + bar.get_height()/2, 
                f'{conf:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_predictions(predictions: list, image_path: Path):
    """Print predictions in formatted output."""
    print(f"\n{'='*60}")
    print(f"EMOTION PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"\nTop {len(predictions)} Predictions:")
    print(f"{'-'*60}")
    
    for i, pred in enumerate(predictions, 1):
        emotion = pred['emotion'].upper()
        confidence = pred['confidence'] * 100
        bar_length = int(confidence / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        print(f"{i}. {emotion:12s} {confidence:6.2f}%  {bar}")
    
    print(f"{'-'*60}")
    print(f"Predicted Emotion: {predictions[0]['emotion'].upper()}")
    print(f"Confidence: {predictions[0]['confidence']*100:.2f}%")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 4: Single Image Emotion Prediction')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['mobilenet', 'efficientnet', 'resnet18', 'resnet50'],
                       help='Model to use for prediction (default: resnet50)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (auto-detected if not provided)')
    parser.add_argument('--detect_face', action='store_true',
                       help='Enable face detection and cropping')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of prediction')
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Save visualization to specified path')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show (default: 3)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return 1
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path('results/checkpoints') / f'{args.model}_finetune_deep_best.pt'
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Please specify checkpoint path with --checkpoint")
        return 1
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, checkpoint_path, device)
    
    # Preprocess image
    print(f"\nPreprocessing image: {image_path.name}")
    img_tensor = preprocess_image(image_path, args.detect_face)
    
    # Predict
    print("Running inference...")
    predictions = predict_emotion(model, img_tensor, device, args.top_k)
    
    # Display results
    print_predictions(predictions, image_path)
    
    # Visualize if requested
    if args.visualize or args.save_viz:
        save_path = Path(args.save_viz) if args.save_viz else None
        visualize_prediction(image_path, predictions, save_path)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
