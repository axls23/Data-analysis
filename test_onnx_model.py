"""Lightweight ONNX Model Inference Script

This script allows testing trained emotion detection models using ONNX Runtime.
No PyTorch installation required - perfect for sharing with teammates!

Dependencies:
    pip install onnxruntime opencv-python pillow numpy

Usage:
    # Basic prediction
    python test_onnx_model.py --image path/to/image.jpg
    
    # Specify different model
    python test_onnx_model.py --image test.jpg --model resnet18
    
    # With face detection
    python test_onnx_model.py --image photo.jpg --detect_face
    
    # Batch process folder
    python test_onnx_model.py --folder test_images/
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json

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
# EfficientNet excluded due to 0% top-1 agreement after quantization
QUANTIZATION_WHITELIST = {
    'mobilenet': True,     # 100% agreement, 72.8% size reduction
    'efficientnet': False, # 0% agreement - broken after quantization
    'resnet18': True,      # 100% agreement, 74.9% size reduction
    'resnet50': True,      # 100% agreement, 74.8% size reduction
}


def detect_face(image_path):
    """Detect and crop face from image using OpenCV Haar Cascade.
    
    Args:
        image_path: Path to input image
        
    Returns:
        PIL.Image: Cropped face image or original if no face detected
    """
    try:
        # Load cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARNING] Could not read image: {image_path}")
            return Image.open(image_path).convert('RGB')
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("[INFO] No face detected, using full image")
            return Image.open(image_path).convert('RGB')
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding (10%)
        padding = int(0.1 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # Crop face
        face_img = img[y1:y2, x1:x2]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        print(f"[INFO] Face detected at ({x}, {y}) with size {w}×{h}")
        return Image.fromarray(face_img)
    
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return Image.open(image_path).convert('RGB')


def preprocess_image(image_path, detect_face_flag=False):
    """Preprocess image for model input.
    
    Args:
        image_path: Path to input image
        detect_face_flag: Whether to detect and crop face
        
    Returns:
        numpy.ndarray: Preprocessed image tensor (1, 3, 224, 224)
    """
    # Load image
    if detect_face_flag:
        img = detect_face(image_path)
    else:
        img = Image.open(image_path).convert('RGB')
    
    # Resize to 224×224
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Convert to array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Transpose to (C, H, W) - channels first
    img_array = img_array.transpose(2, 0, 1)
    
    # Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Add batch dimension (1, 3, 224, 224)
    return img_array[np.newaxis, :].astype(np.float32)


def load_model(model_name, onnx_dir='exported_models/onnx', prefer_quantized=True):
    """Load ONNX model and metadata. Automatically uses quantized version if available and validated.
    
    Args:
        model_name: Name of model (mobilenet, resnet18, resnet50, efficientnet)
        onnx_dir: Directory containing ONNX models
        prefer_quantized: Whether to prefer quantized models (default: True)
        
    Returns:
        tuple: (ort.InferenceSession, dict, bool) - Model session, metadata, and is_quantized flag
    """
    onnx_dir = Path(onnx_dir)
    
    # Determine which model to load based on quantization whitelist
    use_quantized = False
    if prefer_quantized and QUANTIZATION_WHITELIST.get(model_name, False):
        quantized_path = onnx_dir / f'{model_name}_quantized.onnx'
        if quantized_path.exists():
            model_path = quantized_path
            use_quantized = True
            print(f"[INFO] Using quantized model (validated, ~75% smaller)")
        else:
            model_path = onnx_dir / f'{model_name}.onnx'
    else:
        model_path = onnx_dir / f'{model_name}.onnx'
        if model_name == 'efficientnet' and prefer_quantized:
            print(f"[INFO] Quantized version not used (failed validation, 0% agreement)")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"[INFO] Loading ONNX model: {model_path}")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(str(model_path))
    
    # Load metadata if available
    metadata = None
    metadata_path = Path('exported_models/metadata') / f'{model_name}_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"[INFO] Model: {metadata['model_name']}")
        print(f"[INFO] Test Accuracy: {metadata['training_info'].get('final_test_acc', 0)*100:.2f}%")
        
        # Show file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Model Size: {file_size_mb:.2f} MB {'(quantized)' if use_quantized else ''}")
    
    return session, metadata, use_quantized


def predict(session, image_path, detect_face_flag=False, top_k=3):
    """Predict emotion from image.
    
    Args:
        session: ONNX Runtime session
        image_path: Path to input image
        detect_face_flag: Whether to detect and crop face
        top_k: Number of top predictions to show
        
    Returns:
        tuple: (predicted_emotion, confidence, top_k_predictions)
    """
    # Preprocess image
    input_data = preprocess_image(image_path, detect_face_flag)
    
    # Run inference
    outputs = session.run(None, {'input': input_data})
    logits = outputs[0][0]
    
    # Convert to probabilities (softmax)
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Get top prediction
    emotion_idx = np.argmax(probs)
    emotion = EMOTIONS[emotion_idx]
    confidence = probs[emotion_idx]
    
    # Get top-K predictions
    top_k_idx = np.argsort(probs)[-top_k:][::-1]
    top_k_predictions = [(EMOTIONS[idx], probs[idx]) for idx in top_k_idx]
    
    return emotion, confidence, top_k_predictions


def print_results(image_path, emotion, confidence, top_k_predictions):
    """Print formatted prediction results.
    
    Args:
        image_path: Path to input image
        emotion: Predicted emotion
        confidence: Prediction confidence
        top_k_predictions: List of (emotion, confidence) tuples
    """
    print(f"\n{'='*70}")
    print(f"EMOTION PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"Image: {Path(image_path).name}")
    print(f"\nPREDICTED EMOTION: {emotion.upper()}")
    print(f"CONFIDENCE: {confidence*100:.2f}%")
    print(f"\n{'-'*70}")
    print(f"Top-{len(top_k_predictions)} Predictions:")
    print(f"{'-'*70}")
    
    for i, (em, conf) in enumerate(top_k_predictions, 1):
        # Create visual confidence bar
        bar_length = int(conf * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{i}. {em.upper():<12} {conf*100:>6.2f}%  {bar}")
    
    print(f"{'='*70}\n")


def process_single_image(model_name, image_path, detect_face_flag=False):
    """Process a single image.
    
    Args:
        model_name: Name of model to use
        image_path: Path to input image
        detect_face_flag: Whether to detect and crop face
    """
    # Load model (automatically uses quantized if validated)
    session, metadata, is_quantized = load_model(model_name)
    
    # Predict
    emotion, confidence, top_k = predict(session, image_path, detect_face_flag)
    
    # Print results
    print_results(image_path, emotion, confidence, top_k)


def process_folder(model_name, folder_path, detect_face_flag=False, save_csv=False):
    """Process all images in a folder.
    
    Args:
        model_name: Name of model to use
        folder_path: Path to folder containing images
        detect_face_flag: Whether to detect and crop face
        save_csv: Whether to save results to CSV
    """
    folder_path = Path(folder_path)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"[ERROR] No images found in {folder_path}")
        return
    
    print(f"\n[INFO] Found {len(image_files)} images to process")
    
    # Load model once (automatically uses quantized if validated)
    session, metadata, is_quantized = load_model(model_name)
    
    # Process each image
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
            
            emotion, confidence, top_k = predict(session, image_path, detect_face_flag)
            
            # Store result
            results.append({
                'image': image_path.name,
                'predicted_emotion': emotion,
                'confidence': confidence,
                'top3': top_k
            })
            
            # Print result
            print(f"  → {emotion.upper()} ({confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(results)}/{len(image_files)}")
    
    # Emotion distribution
    emotion_counts = {}
    for result in results:
        em = result['predicted_emotion']
        emotion_counts[em] = emotion_counts.get(em, 0) + 1
    
    print(f"\nEmotion Distribution:")
    print(f"{'-'*70}")
    for emotion in EMOTIONS:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / len(results) * 100) if results else 0
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{emotion.upper():<12} {count:>3} ({percentage:>5.1f}%)  {bar}")
    
    print(f"{'='*70}\n")
    
    # Save to CSV if requested
    if save_csv:
        import csv
        csv_path = folder_path / f'{model_name}_predictions.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Predicted_Emotion', 'Confidence', 'Top3_Predictions'])
            
            for result in results:
                top3_str = " | ".join([f"{em}: {conf*100:.1f}%" 
                                      for em, conf in result['top3']])
                writer.writerow([
                    result['image'],
                    result['predicted_emotion'],
                    f"{result['confidence']*100:.2f}%",
                    top3_str
                ])
        
        print(f"[INFO] Results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight ONNX Emotion Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image prediction
    python test_onnx_model.py --image test.jpg
    
    # With face detection
    python test_onnx_model.py --image photo.jpg --detect_face
    
    # Use different model
    python test_onnx_model.py --image test.jpg --model resnet18
    
    # Batch process folder
    python test_onnx_model.py --folder test_images/ --save_csv
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                            help='Path to single input image')
    input_group.add_argument('--folder', type=str,
                            help='Path to folder containing images to process')
    
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=AVAILABLE_MODELS,
                       help='Model to use for prediction (default: resnet50)')
    parser.add_argument('--detect_face', action='store_true',
                       help='Enable face detection and cropping')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save batch results to CSV (folder mode only)')
    parser.add_argument('--onnx_dir', type=str, default='exported_models/onnx',
                       help='Directory containing ONNX models')
    
    args = parser.parse_args()
    
    # Process
    try:
        if args.image:
            # Single image mode
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"[ERROR] Image not found: {image_path}")
                return 1
            
            process_single_image(args.model, image_path, args.detect_face)
        
        else:
            # Folder mode
            folder_path = Path(args.folder)
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"[ERROR] Folder not found: {folder_path}")
                return 1
            
            process_folder(args.model, folder_path, args.detect_face, args.save_csv)
        
        return 0
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
