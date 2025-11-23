"""Phase 4: Real-Time Webcam Emotion Detection Demo

Live emotion detection using webcam with face detection and model inference.

Features:
- Real-time webcam capture with OpenCV
- Face detection using Haar Cascade
- Emotion prediction with confidence scores
- Visual overlay with bounding boxes and labels
- FPS counter and performance metrics
- Multiple model support

Controls:
    - Press 'q' to quit
    - Press 's' to save screenshot
    - Press 'f' to toggle face detection overlay
    - Press 'm' to cycle through models (if multiple loaded)

Usage:
    # Basic demo
    python webcam_demo.py --model resnet50
    
    # Custom checkpoint
    python webcam_demo.py --model resnet18 --checkpoint results/checkpoints/resnet18_finetune_deep_best.pt
    
    # Adjust confidence threshold
    python webcam_demo.py --model resnet50 --threshold 0.5
    
    # Save screenshots to custom directory
    python webcam_demo.py --model mobilenet --save_dir demo_screenshots
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
from collections import deque
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import EMOTION_LABELS, USE_GPU, GPU_ID, IMAGE_SIZE
from models import create_model


# Emotion colors (BGR format for OpenCV)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 0),    # Dark Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 0),      # Green
    'neutral': (128, 128, 128),# Gray
    'sad': (255, 0, 0),        # Blue
    'surprised': (0, 255, 255) # Yellow
}


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
    
    model = create_model(model_name, num_classes=7, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model


class EmotionDetector:
    """Real-time emotion detector with webcam."""
    
    def __init__(self, model, device, confidence_threshold=0.3):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Setup transforms (validation mode, no augmentation)
        self.transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        
        # Settings
        self.show_face_box = True
        self.screenshot_counter = 0
    
    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input."""
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        img_tensor = self.transforms(pil_img)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def predict_emotion(self, face_tensor):
        """Predict emotion from preprocessed face tensor."""
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get top prediction
        confidence, pred_idx = torch.max(probs, dim=1)
        emotion = EMOTION_LABELS[pred_idx.item()]
        
        return emotion, confidence.item(), probs[0].cpu().numpy()
    
    def draw_overlay(self, frame, faces, predictions, fps):
        """Draw bounding boxes, labels, and info overlay on frame."""
        h, w = frame.shape[:2]
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to screenshot",
            "Press 'f' to toggle boxes"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, h - 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw faces and predictions
        for (x, y, face_w, face_h), (emotion, confidence, probs) in zip(faces, predictions):
            # Skip low confidence
            if confidence < self.confidence_threshold:
                continue
            
            # Get emotion color
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            # Draw bounding box
            if self.show_face_box:
                cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), color, 2)
            
            # Prepare label text
            label = f'{emotion.upper()}: {confidence*100:.1f}%'
            
            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_width = 150
            bar_height = 20
            bar_x = x
            bar_y = y + face_h + 10
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Confidence bar
            filled_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         color, -1)
            
            # Bar border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 1)
            
            # Top-3 emotions (small text below bar)
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_text = ' | '.join([f'{EMOTION_LABELS[i]}: {probs[i]*100:.0f}%' 
                                   for i in top3_indices])
            cv2.putText(frame, top3_text, (bar_x, bar_y + bar_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Draw emotion count
        if predictions:
            emotion_counts = {}
            for _, (emotion, conf, _) in zip(faces, predictions):
                if conf >= self.confidence_threshold:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if emotion_counts:
                y_offset = 60
                cv2.putText(frame, 'Detected:', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                for emotion, count in sorted(emotion_counts.items()):
                    y_offset += 20
                    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                    cv2.putText(frame, f'{emotion}: {count}', (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def save_screenshot(self, frame, save_dir):
        """Save current frame as screenshot."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'emotion_demo_{timestamp}_{self.screenshot_counter:03d}.jpg'
        filepath = save_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"Screenshot saved: {filepath}")
        
        self.screenshot_counter += 1
    
    def run(self, camera_id=0, save_dir='screenshots'):
        """Run webcam demo loop."""
        print(f"\nStarting webcam demo...")
        print(f"Camera ID: {camera_id}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"\nControls:")
        print(f"  'q' - Quit")
        print(f"  's' - Save screenshot")
        print(f"  'f' - Toggle face boxes")
        print()
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_id}")
            return 1
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam opened successfully. Press 'q' to quit.\n")
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                predictions = []
                for (x, y, w, h) in faces:
                    # Add padding
                    padding = int(0.1 * max(w, h))
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    # Extract face
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Preprocess and predict
                    face_tensor = self.preprocess_face(face_img)
                    emotion, confidence, probs = self.predict_emotion(face_tensor)
                    
                    predictions.append((emotion, confidence, probs))
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)
                
                # Draw overlay
                display_frame = self.draw_overlay(frame, faces, predictions, avg_fps)
                
                # Show frame
                cv2.imshow('Emotion Detection Demo', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self.save_screenshot(display_frame, save_dir)
                elif key == ord('f'):
                    self.show_face_box = not self.show_face_box
                    print(f"Face boxes: {'ON' if self.show_face_box else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam demo ended.")
        
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 4: Real-Time Webcam Emotion Detection')
    
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['mobilenet', 'efficientnet', 'resnet18', 'resnet50'],
                       help='Model to use for prediction (default: resnet50)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (auto-detected if not provided)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID to use (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Confidence threshold for display (default: 0.3)')
    parser.add_argument('--save_dir', type=str, default='screenshots',
                       help='Directory to save screenshots (default: screenshots)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    # Create detector
    detector = EmotionDetector(model, device, args.threshold)
    
    # Run demo
    return detector.run(args.camera, args.save_dir)


if __name__ == '__main__':
    raise SystemExit(main())
