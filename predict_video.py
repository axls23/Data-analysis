"""Phase 5: Video Emotion Detection and Analysis

Process videos to detect and track emotions over time.

Features:
- Frame-by-frame emotion detection
- Multi-face tracking with unique IDs
- Temporal smoothing for stable predictions
- Emotion timeline visualization
- Annotated output video with overlays
- CSV export with timestamps
- Summary statistics and emotion distribution

Usage:
    python predict_video.py --video input.mp4 --model resnet50
    python predict_video.py --video demo.mp4 --model resnet18 --skip_frames 2 --output_video annotated.mp4
    python predict_video.py --video webcam.avi --model mobilenet --smooth_window 5 --save_timeline
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import json
from collections import deque, defaultdict
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import EMOTION_LABELS, USE_GPU, GPU_ID
from models import create_model


AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']

# Emotion colors (BGR for OpenCV)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 0),    # Dark Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 255),    # Yellow
    'neutral': (128, 128, 128), # Gray
    'sad': (255, 0, 0),        # Blue
    'surprised': (255, 165, 0) # Orange
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
    print(f"Loading {model_name} from {checkpoint_path}")
    
    model = create_model(model_name, num_classes=7, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model = model.to(device)
    
    return model


def get_transform():
    """Get image preprocessing transforms."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


class TemporalSmoother:
    """Smooth predictions over time using moving average."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def update(self, probabilities: np.ndarray) -> np.ndarray:
        """Add new prediction and return smoothed result."""
        self.history.append(probabilities)
        return np.mean(self.history, axis=0)
    
    def reset(self):
        """Reset history."""
        self.history.clear()


class FaceTracker:
    """Track faces across frames using simple IoU matching."""
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.active_tracks = {}  # {track_id: bbox}
        self.smoothers = {}  # {track_id: TemporalSmoother}
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, face_boxes: list) -> dict:
        """Update tracks with new detections. Returns {track_id: bbox}."""
        if not face_boxes:
            self.active_tracks.clear()
            return {}
        
        # Match detections to existing tracks
        matched_tracks = {}
        unmatched_detections = list(face_boxes)
        
        for track_id, prev_bbox in self.active_tracks.items():
            best_iou = 0.0
            best_match = None
            
            for bbox in unmatched_detections:
                iou = self._compute_iou(prev_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = bbox
            
            if best_iou > self.iou_threshold and best_match is not None:
                matched_tracks[track_id] = best_match
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for bbox in unmatched_detections:
            matched_tracks[self.next_id] = bbox
            self.smoothers[self.next_id] = TemporalSmoother()
            self.next_id += 1
        
        self.active_tracks = matched_tracks
        return matched_tracks
    
    def get_smoother(self, track_id: int) -> TemporalSmoother:
        """Get temporal smoother for a track."""
        if track_id not in self.smoothers:
            self.smoothers[track_id] = TemporalSmoother()
        return self.smoothers[track_id]


def detect_faces_haar(frame):
    """Detect faces using Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def predict_emotion(face_img, model, transform, device):
    """Predict emotion for a single face image."""
    # Preprocess
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(face_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    emotion_idx = probabilities.argmax()
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = probabilities[emotion_idx]
    
    return emotion, confidence, probabilities


def draw_emotion_overlay(frame, bbox, track_id, emotion, confidence):
    """Draw emotion label and bounding box on frame."""
    x, y, w, h = bbox
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw label background
    label = f"ID{track_id}: {emotion.upper()} ({confidence:.2f})"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def create_emotion_timeline(timeline_data: list, output_path: Path):
    """Create emotion timeline visualization."""
    if not timeline_data:
        return
    
    # Extract data
    frames = [d['frame'] for d in timeline_data]
    timestamps = [d['timestamp'] for d in timeline_data]
    
    # Count emotions per frame
    emotion_counts = {emotion: [] for emotion in EMOTION_LABELS}
    
    for frame_data in timeline_data:
        frame_emotions = defaultdict(int)
        for face in frame_data['faces']:
            frame_emotions[face['emotion']] += 1
        
        for emotion in EMOTION_LABELS:
            emotion_counts[emotion].append(frame_emotions[emotion])
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for emotion in EMOTION_LABELS:
        if max(emotion_counts[emotion]) > 0:
            color_bgr = EMOTION_COLORS[emotion]
            color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
            ax.plot(timestamps, emotion_counts[emotion], label=emotion.capitalize(), color=color_rgb, linewidth=2)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Number of Faces', fontsize=12)
    ax.set_title('Emotion Detection Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[TIMELINE] Saved to: {output_path}")


def process_video(
    video_path: Path,
    model_name: str,
    checkpoint_dir: Path,
    output_dir: Path,
    skip_frames: int = 0,
    smooth_window: int = 5,
    output_video: bool = True,
    save_timeline: bool = True
):
    """Process video and detect emotions."""
    print(f"\n{'='*70}")
    print(f"VIDEO EMOTION DETECTION")
    print(f"{'='*70}")
    print(f"Input video: {video_path}")
    print(f"Model: {model_name}")
    print(f"Skip frames: {skip_frames} (process every {skip_frames + 1} frames)")
    print(f"Smoothing window: {smooth_window} frames")
    print(f"{'='*70}\n")
    
    # Setup
    device = get_device()
    checkpoint_path = checkpoint_dir / f'{model_name}_finetune_deep_best.pt'
    model = load_model(model_name, checkpoint_path, device)
    transform = get_transform()
    face_tracker = FaceTracker()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames ({duration:.1f}s)")
    
    # Setup output video writer
    video_writer = None
    if output_video:
        output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"Output video: {output_video_path}")
    
    # Setup CSV writer
    csv_path = output_dir / f"{video_path.stem}_emotions.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'timestamp', 'track_id', 'emotion', 'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
    
    # Timeline data
    timeline_data = []
    
    # Emotion statistics
    emotion_stats = defaultdict(int)
    total_detections = 0
    
    # Process frames
    frame_idx = 0
    processed_frames = 0
    
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_idx / fps
        
        # Skip frames if requested
        if frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            pbar.update(1)
            continue
        
        # Detect faces
        faces = detect_faces_haar(frame)
        
        # Update tracker
        tracks = face_tracker.update(faces.tolist() if len(faces) > 0 else [])
        
        # Process each tracked face
        frame_faces_data = []
        
        for track_id, (x, y, w, h) in tracks.items():
            # Extract face
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
            
            # Predict emotion
            emotion, confidence, probabilities = predict_emotion(face_img, model, transform, device)
            
            # Apply temporal smoothing
            smoother = face_tracker.get_smoother(track_id)
            smoothed_probs = smoother.update(probabilities)
            smoothed_emotion_idx = smoothed_probs.argmax()
            smoothed_emotion = EMOTION_LABELS[smoothed_emotion_idx]
            smoothed_confidence = smoothed_probs[smoothed_emotion_idx]
            
            # Update statistics
            emotion_stats[smoothed_emotion] += 1
            total_detections += 1
            
            # Draw overlay
            draw_emotion_overlay(frame, (x, y, w, h), track_id, smoothed_emotion, smoothed_confidence)
            
            # Record data
            csv_writer.writerow([frame_idx, f"{timestamp:.2f}", track_id, smoothed_emotion, f"{smoothed_confidence:.4f}", x, y, w, h])
            frame_faces_data.append({
                'track_id': track_id,
                'emotion': smoothed_emotion,
                'confidence': float(smoothed_confidence),
                'bbox': [x, y, w, h]
            })
        
        # Record timeline
        timeline_data.append({
            'frame': frame_idx,
            'timestamp': timestamp,
            'faces': frame_faces_data
        })
        
        # Write annotated frame
        if video_writer:
            video_writer.write(frame)
        
        frame_idx += 1
        processed_frames += 1
        pbar.update(skip_frames + 1)
    
    pbar.close()
    cap.release()
    if video_writer:
        video_writer.release()
    csv_file.close()
    
    print(f"\n[CSV] Saved to: {csv_path}")
    
    # Save timeline visualization
    if save_timeline and timeline_data:
        timeline_path = output_dir / f"{video_path.stem}_timeline.png"
        create_emotion_timeline(timeline_data, timeline_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total face detections: {total_detections}")
    print(f"\nEmotion Distribution:")
    for emotion in EMOTION_LABELS:
        count = emotion_stats[emotion]
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {emotion.capitalize():12s}: {count:4d} ({percentage:5.1f}%)")
    
    # Save summary JSON
    summary_path = output_dir / f"{video_path.stem}_summary.json"
    summary = {
        'video_info': {
            'path': str(video_path),
            'duration_seconds': duration,
            'fps': fps,
            'total_frames': total_frames,
            'resolution': f"{width}x{height}"
        },
        'processing_info': {
            'model': model_name,
            'frames_processed': processed_frames,
            'skip_frames': skip_frames,
            'smooth_window': smooth_window
        },
        'statistics': {
            'total_detections': total_detections,
            'emotion_distribution': dict(emotion_stats),
            'emotion_percentages': {
                emotion: (count / total_detections * 100) if total_detections > 0 else 0
                for emotion, count in emotion_stats.items()
            }
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUMMARY] Saved to: {summary_path}")
    print(f"\n{'='*70}")
    print("VIDEO PROCESSING COMPLETE!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Video emotion detection and analysis')
    parser.add_argument('--video', type=Path, required=True,
                       help='Input video file path')
    parser.add_argument('--model', type=str, default='resnet50', choices=AVAILABLE_MODELS,
                       help='Model to use for prediction')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path('results/checkpoints'),
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=Path, default=None,
                       help='Output directory (default: video_results/<video_name>)')
    parser.add_argument('--skip_frames', type=int, default=0,
                       help='Process every Nth frame (0 = process all frames)')
    parser.add_argument('--smooth_window', type=int, default=5,
                       help='Temporal smoothing window size')
    parser.add_argument('--no_output_video', action='store_true',
                       help='Skip creating annotated output video')
    parser.add_argument('--save_timeline', action='store_true',
                       help='Generate emotion timeline visualization')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path('video_results') / args.video.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video
    process_video(
        args.video,
        args.model,
        args.checkpoint_dir,
        args.output_dir,
        args.skip_frames,
        args.smooth_window,
        not args.no_output_video,
        args.save_timeline
    )


if __name__ == '__main__':
    main()
