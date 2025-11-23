"""Compare PyTorch vs ONNX predictions to verify export correctness"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image

from models import create_model
from utils.data_loader import get_transforms

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

def load_pytorch_model():
    """Load PyTorch model from checkpoint"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model('resnet50', num_classes=7, pretrained=False)
    checkpoint = torch.load('results/checkpoints/resnet50_finetune_deep_best.pt', 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model, device

def load_onnx_model():
    """Load ONNX model"""
    session = ort.InferenceSession('exported_models/onnx/resnet50.onnx')
    return session

def preprocess_pytorch(image_path, device):
    """Preprocess for PyTorch model"""
    transforms = get_transforms('val')
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms(img).unsqueeze(0).to(device)
    return img_tensor

def preprocess_onnx(image_path):
    """Preprocess for ONNX model"""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    return img_array[np.newaxis, :].astype(np.float32)

def compare_predictions(image_path):
    """Compare PyTorch and ONNX predictions"""
    print(f"\n{'='*70}")
    print(f"Comparing predictions for: {Path(image_path).name}")
    print(f"{'='*70}")
    
    # Load models
    pytorch_model, device = load_pytorch_model()
    onnx_session = load_onnx_model()
    
    # PyTorch prediction
    pytorch_input = preprocess_pytorch(image_path, device)
    with torch.no_grad():
        pytorch_logits = pytorch_model(pytorch_input).cpu().numpy()[0]
    
    pytorch_probs = torch.softmax(torch.from_numpy(pytorch_logits), dim=0).numpy()
    pytorch_pred = EMOTIONS[np.argmax(pytorch_probs)]
    pytorch_conf = pytorch_probs[np.argmax(pytorch_probs)]
    
    # ONNX prediction
    onnx_input = preprocess_onnx(image_path)
    onnx_logits = onnx_session.run(None, {'input': onnx_input})[0][0]
    
    onnx_probs = np.exp(onnx_logits) / np.sum(np.exp(onnx_logits))
    onnx_pred = EMOTIONS[np.argmax(onnx_probs)]
    onnx_conf = onnx_probs[np.argmax(onnx_probs)]
    
    # Print results
    print(f"\nPyTorch Model:")
    print(f"  Prediction: {pytorch_pred.upper()} ({pytorch_conf*100:.2f}%)")
    print(f"  Probabilities: {', '.join([f'{e}: {p*100:.1f}%' for e, p in zip(EMOTIONS, pytorch_probs)])}")
    
    print(f"\nONNX Model:")
    print(f"  Prediction: {onnx_pred.upper()} ({onnx_conf*100:.2f}%)")
    print(f"  Probabilities: {', '.join([f'{e}: {p*100:.1f}%' for e, p in zip(EMOTIONS, onnx_probs)])}")
    
    # Check differences
    print(f"\n{'='*70}")
    print("Comparison:")
    print(f"{'='*70}")
    print(f"Predictions match: {pytorch_pred == onnx_pred}")
    print(f"Max probability diff: {np.abs(pytorch_probs - onnx_probs).max():.6f}")
    print(f"Max logit diff: {np.abs(pytorch_logits - onnx_logits).max():.6f}")
    
    if pytorch_pred != onnx_pred:
        print(f"\n⚠️  WARNING: Predictions don't match!")
        print(f"PyTorch: {pytorch_pred} vs ONNX: {onnx_pred}")
    else:
        print(f"\n✓ Predictions match!")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_images/test1.jpg'
    compare_predictions(image_path)
