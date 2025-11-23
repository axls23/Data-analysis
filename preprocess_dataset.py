"""
Dataset Preprocessing Pipeline
================================

This script implements Phase 1 of the Facial Emotion Detection project:
1. Data Cleaning - Validate and remove corrupted images
2. Face Detection & Cropping - Extract face regions using MTCNN
3. Resize - Standardize all images to 224x224 pixels
4. Quality Control - Verify face detection quality

Usage:
    python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data
    python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data --margin 0.3
    python preprocess_dataset.py --help

Author: AI Assistant
Date: November 2025
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import torch

# MTCNN for face detection
try:
    from facenet_pytorch import MTCNN
    
except ImportError:
    raise ImportError(
        "MTCNN is required for preprocessing. Install with: pip install facenet-pytorch"
    )


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


class FaceDetector:
    """MTCNN-based face detector for high-quality face detection"""
    
    def __init__(self, device_info=None):
        """
        Initialize MTCNN face detector
        
        Args:
            device_info: Device information dict with 'device' key
        """
        self.device = device_info['device'] if device_info else 'cpu'
        
        self.detector = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            post_process=False,
            device=self.device
        )
    
    def detect_face(self, image):
        """
        Detect face in image using MTCNN
        
        Args:
            image: PIL Image or numpy array (BGR)
        
        Returns:
            success: Boolean indicating if face was detected
            face_box: (x, y, w, h) bounding box or None
            confidence: Detection confidence (0-1) or None
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return self._detect_mtcnn(image)
    
    def _detect_mtcnn(self, image):
        """Detect face using MTCNN"""
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Detect face
        boxes, probs = self.detector.detect(pil_image)
        
        if boxes is not None and len(boxes) > 0:
            # Get first (most confident) face
            box = boxes[0]
            prob = probs[0]
            
            # Convert to (x, y, w, h) format
            x, y, x2, y2 = box.astype(int)
            w, h = x2 - x, y2 - y
            
            return True, (x, y, w, h), float(prob)
        
        return False, None, None


def validate_image(image_path):
    """
    Validate that image can be loaded and is not corrupted
    
    Args:
        image_path: Path to image file
    
    Returns:
        valid: Boolean indicating if image is valid
        error: Error message if invalid, None otherwise
    """
    try:
        # Try loading with PIL
        img = Image.open(image_path)
        img.verify()  # Verify it's a valid image
        
        # Try loading again (verify() closes the file)
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB format
        
        # Check minimum size
        if img.size[0] < 50 or img.size[1] < 50:
            return False, "Image too small (< 50x50)"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def crop_face_with_margin(image, face_box, margin=0.2, target_size=(224, 224)):
    """
    Crop face from image with margin and resize
    
    Args:
        image: OpenCV image (BGR)
        face_box: (x, y, w, h) face bounding box
        margin: Margin to add around face (as fraction of face size)
        target_size: Target size for output image
    
    Returns:
        cropped_face: Cropped and resized face image
    """
    x, y, w, h = face_box
    img_h, img_w = image.shape[:2]
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    # Calculate crop box with margin
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)
    
    # Crop face
    face = image[y1:y2, x1:x2]
    
    # Resize to target size
    face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return face_resized


def preprocess_image(image_path, output_path, detector, margin=0.2, target_size=(224, 224)):
    """
    Complete preprocessing pipeline for a single image
    
    Args:
        image_path: Input image path
        output_path: Output image path
        detector: FaceDetector instance
        margin: Margin around face
        target_size: Target size for output
    
    Returns:
        success: Boolean indicating success
        message: Status message
        confidence: Detection confidence (if successful)
    """
    # Validate image
    valid, error = validate_image(image_path)
    if not valid:
        return False, f"Invalid image: {error}", None
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return False, "Failed to load image with OpenCV", None
    
    # Detect face
    success, face_box, confidence = detector.detect_face(image)
    if not success:
        return False, "No face detected", None
    
    # Crop and resize face
    try:
        face_cropped = crop_face_with_margin(image, face_box, margin, target_size)
    except Exception as e:
        return False, f"Failed to crop face: {str(e)}", None
    
    # Save preprocessed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), face_cropped)
    
    return True, "Success", confidence


def process_dataset(input_dir, output_dir, margin=0.2, 
                    target_size=(224, 224), skip_existing=False, device_info=None):
    """
    Process entire dataset using MTCNN face detection
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for preprocessed images
        margin: Margin around detected face
        target_size: Target image size
        skip_existing: Skip already processed images
        device_info: Device information dict
    
    Returns:
        stats: Dictionary with processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Initialize MTCNN detector
    print(f"[INFO] Initializing MTCNN face detector on {device_info['name']}...")
    detector = FaceDetector(device_info=device_info)
    
    # Find all images in emotion folders
    emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    all_images = []
    for dataset_folder in sorted(input_path.glob('d*')):
        if not dataset_folder.is_dir():
            continue
        
        for emotion in emotion_folders:
            emotion_path = dataset_folder / emotion
            if not emotion_path.exists():
                continue
            
            for img_path in emotion_path.glob('*'):
                if img_path.suffix.lower() in image_extensions:
                    # Maintain relative structure in output
                    rel_path = img_path.relative_to(input_path)
                    out_path = output_path / rel_path
                    
                    all_images.append((img_path, out_path, emotion))
    
    if len(all_images) == 0:
        print("[ERROR] No images found in dataset!")
        return None
    
    print(f"[INFO] Found {len(all_images)} images to process")
    
    # Process images
    stats = {
        'total': len(all_images),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'errors': {},
        'confidences': [],
        'emotion_counts': {emotion: {'total': 0, 'success': 0, 'failed': 0} 
                          for emotion in emotion_folders}
    }
    
    for img_path, out_path, emotion in tqdm(all_images, desc="Processing images"):
        # Skip if already exists
        if skip_existing and out_path.exists():
            stats['skipped'] += 1
            continue
        
        # Preprocess
        success, message, confidence = preprocess_image(
            img_path, out_path, detector, margin, target_size
        )
        
        # Update stats
        stats['emotion_counts'][emotion]['total'] += 1
        
        if success:
            stats['success'] += 1
            stats['emotion_counts'][emotion]['success'] += 1
            if confidence is not None:
                stats['confidences'].append(confidence)
        else:
            stats['failed'] += 1
            stats['emotion_counts'][emotion]['failed'] += 1
            
            # Track error types
            error_type = message.split(':')[0] if ':' in message else message
            if error_type not in stats['errors']:
                stats['errors'][error_type] = []
            stats['errors'][error_type].append(str(img_path))
    
    # Save statistics
    stats_file = output_path / 'preprocessing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total': stats['total'],
            'success': stats['success'],
            'failed': stats['failed'],
            'skipped': stats['skipped'],
            'success_rate': stats['success'] / stats['total'] if stats['total'] > 0 else 0,
            'avg_confidence': np.mean(stats['confidences']) if stats['confidences'] else 0,
            'emotion_counts': stats['emotion_counts'],
            'error_summary': {k: len(v) for k, v in stats['errors'].items()}
        }, f, indent=2)
    
    print(f"\n[INFO] Statistics saved to {stats_file}")
    
    return stats


def print_statistics(stats):
    """Print processing statistics"""
    if stats is None:
        return
    
    print("\n" + "="*70)
    print("PREPROCESSING STATISTICS")
    print("="*70)
    print(f"Total images:        {stats['total']}")
    print(f"Successfully processed: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Failed:              {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    if stats['skipped'] > 0:
        print(f"Skipped (existing):  {stats['skipped']}")
    
    if stats['confidences']:
        print(f"\nAverage confidence:  {np.mean(stats['confidences']):.3f}")
    
    print("\nPer-Emotion Statistics:")
    print("-"*70)
    for emotion, counts in stats['emotion_counts'].items():
        if counts['total'] > 0:
            success_rate = counts['success'] / counts['total'] * 100
            print(f"  {emotion:12s}: {counts['success']:4d}/{counts['total']:4d} ({success_rate:5.1f}%)")
    
    if stats['errors']:
        print("\nError Summary:")
        print("-"*70)
        for error_type, files in stats['errors'].items():
            print(f"  {error_type}: {len(files)} images")
            if len(files) <= 5:
                for f in files:
                    print(f"    - {f}")
            else:
                print(f"    (showing first 3)")
                for f in files[:3]:
                    print(f"    - {f}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess facial emotion detection dataset using MTCNN - Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data
  
  # Custom margin and target size
  python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data --margin 0.3 --size 256
  
  # Skip already processed images
  python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data --skip_existing
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='dataset',
                       help='Input dataset directory (default: dataset)')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                       help='Output directory for preprocessed images (default: preprocessed_data)')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Margin around detected face (default: 0.2)')
    parser.add_argument('--size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip already preprocessed images')
    
    args = parser.parse_args()
    
    # Get device information
    device_info = get_device_info()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] Input directory does not exist: {args.input_dir}")
        return 1
    
    print("="*70)
    print("FACIAL EMOTION DETECTION - DATASET PREPROCESSING (MTCNN)")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Face detector:    MTCNN")
    print(f"Compute device:   {device_info['name']}")
    if device_info['gpu_available']:
        print(f"GPU available:    Yes ({device_info['device_count']} GPU(s))")
    print(f"Margin:           {args.margin}")
    print(f"Target size:      {args.size}x{args.size}")
    print(f"Skip existing:    {args.skip_existing}")
    print("="*70 + "\n")
    
    # Process dataset
    stats = process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        margin=args.margin,
        target_size=(args.size, args.size),
        skip_existing=args.skip_existing,
        device_info=device_info
    )
    
    # Print statistics
    print_statistics(stats)
    
    if stats and stats['success'] > 0:
        print(f"\n✓ Preprocessing complete! Preprocessed images saved to: {args.output_dir}/")
        return 0
    else:
        print("\n✗ Preprocessing failed or no images processed.")
        return 1


if __name__ == '__main__':
    exit(main())
