# Facial Emotion Detection - Exported Models

This folder contains trained emotion detection models in production-ready formats (ONNX and TorchScript).

## Quick Start (No PyTorch Required!)

### 1. Install Dependencies

```bash
pip install onnxruntime opencv-python pillow numpy
```

**That's it!** No need to install the full PyTorch stack.

### 2. Test the Model

```bash
# Single image prediction
python test_onnx_model.py --image path/to/your/image.jpg

# With face detection (recommended for photos with backgrounds)
python test_onnx_model.py --image photo.jpg --detect_face

# Process entire folder
python test_onnx_model.py --folder test_images/ --save_csv
```

### 3. Example Output

```
======================================================================
EMOTION PREDICTION RESULT
======================================================================
Image: happy_face.jpg

PREDICTED EMOTION: HAPPY
CONFIDENCE: 87.34%

----------------------------------------------------------------------
Top-3 Predictions:
----------------------------------------------------------------------
1. HAPPY        87.34%  ███████████████████████████████████████████░░░
2. SURPRISED    08.21%  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
3. NEUTRAL      02.15%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
======================================================================
```

---

## Available Models

| Model | Test Accuracy | File Size | Inference Speed | Best For |
|-------|--------------|-----------|-----------------|----------|
| **resnet50** | **74.4%** | 90 MB | ~6-8 ms | **Highest accuracy** (recommended) |
| **resnet18** | **68.9%** | 45 MB | ~4-5 ms | Good balance |
| **mobilenet** | **60.7%** | 9 MB | ~1-2 ms | Fastest inference, mobile devices |
| **efficientnet** | **57.6%** | 17 MB | ~3-4 ms | Good efficiency |

**Recommendation**: Use **ResNet50** for best results. Use **MobileNet** if speed is critical.

---

## File Structure

```
exported_models/
├── onnx/                          # ONNX models (cross-platform)
│   ├── resnet50.onnx             # 74.4% accuracy (recommended)
│   ├── resnet18.onnx             # 68.9% accuracy
│   ├── mobilenet.onnx            # 60.7% accuracy (fastest)
│   └── efficientnet.onnx         # 57.6% accuracy
├── torchscript/                   # TorchScript models (PyTorch/C++)
│   ├── resnet50_traced.pt
│   ├── resnet18_traced.pt
│   ├── mobilenet_traced.pt
│   └── efficientnet_traced.pt
├── metadata/                      # Model metadata & preprocessing info
│   ├── resnet50_metadata.json
│   ├── resnet18_metadata.json
│   ├── mobilenet_metadata.json
│   └── efficientnet_metadata.json
└── README.md                      # This file
```

---

## Usage Examples

### Single Image Prediction

```bash
# Basic prediction with ResNet50 (best model)
python test_onnx_model.py --image happy_person.jpg

# Use different model
python test_onnx_model.py --image test.jpg --model resnet18

# With face detection (crops face automatically)
python test_onnx_model.py --image group_photo.jpg --detect_face
```

### Batch Processing

```bash
# Process entire folder
python test_onnx_model.py --folder my_images/

# Save results to CSV
python test_onnx_model.py --folder test_images/ --save_csv

# With face detection for all images
python test_onnx_model.py --folder photos/ --detect_face --save_csv
```

---

## Detected Emotions

The models can detect 7 emotions:
1. **Angry** - Furrowed brows, tense jaw
2. **Disgust** - Wrinkled nose, raised upper lip
3. **Fear** - Widened eyes, open mouth
4. **Happy** - Smile, raised cheeks
5. **Neutral** - Relaxed, no strong expression
6. **Sad** - Lowered eyebrows, downturned mouth
7. **Surprised** - Raised eyebrows, open mouth, wide eyes

---

## Input Requirements

### Image Specifications
- **Format**: JPG, PNG, BMP, TIFF, WebP
- **Resolution**: Any (automatically resized to 224×224)
- **Color**: RGB (grayscale will be converted)

### Best Practices
✅ **Use face detection** (`--detect_face`) for photos with backgrounds  
✅ **Good lighting** improves accuracy  
✅ **Front-facing faces** work best  
✅ **Clear expressions** yield better results  
❌ **Avoid** very blurry or low-resolution images  
❌ **Avoid** extreme angles or partial faces  

---

## Advanced Usage

### Python Integration

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("exported_models/onnx/resnet50.onnx")

# Preprocess image (resize to 224×224, normalize)
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    return img_array[np.newaxis, :].astype(np.float32)

# Run inference
input_data = preprocess("test.jpg")
outputs = session.run(None, {'input': input_data})
probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
predicted_emotion = emotions[np.argmax(probabilities)]
print(f"Predicted: {predicted_emotion}")
```

### C++ Integration (TorchScript)

```cpp
#include <torch/script.h>
#include <opencv2/opencv.hpp>

int main() {
    // Load model
    torch::jit::script::Module module;
    module = torch::jit::load("exported_models/torchscript/resnet50_traced.pt");
    
    // Load and preprocess image
    cv::Mat image = cv::imread("test.jpg");
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    
    // Normalize with ImageNet stats
    // ... (see full example in C++ docs)
    
    // Run inference
    auto output = module.forward({input_tensor}).toTensor();
    int emotion_idx = output.argmax(1).item<int>();
    
    return 0;
}
```

---

## Deployment Platforms

### Supported Platforms

**ONNX Format** (cross-platform):
- ✅ Python (Windows, Linux, macOS)
- ✅ C++ / C#
- ✅ Java / JavaScript (ONNX.js for web)
- ✅ Mobile (iOS CoreML, Android NNAPI)
- ✅ Edge devices (Raspberry Pi, Jetson Nano)

**TorchScript Format** (PyTorch ecosystem):
- ✅ Python (PyTorch)
- ✅ C++ (LibTorch)
- ✅ Mobile (PyTorch Mobile)

---

## Performance Benchmarks

**Inference Speed** (NVIDIA RTX 4060, single image):

| Model | PyTorch | TorchScript | ONNX Runtime |
|-------|---------|-------------|--------------|
| ResNet50 | 8.2 ms | 7.1 ms | **6.5 ms** ⚡ |
| ResNet18 | 5.4 ms | 4.8 ms | **4.2 ms** ⚡ |
| MobileNet | 3.1 ms | 2.7 ms | **2.3 ms** ⚡ |
| EfficientNet | 4.6 ms | 4.0 ms | **3.5 ms** ⚡ |

**Real-time FPS** (single face detection + classification):
- MobileNet: ~320 FPS ⚡
- EfficientNet: ~220 FPS
- ResNet18: ~185 FPS
- ResNet50: ~120 FPS

---

## Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'onnxruntime'`  
**Solution**: 
```bash
pip install onnxruntime
```

**Problem**: `ModuleNotFoundError: No module named 'cv2'`  
**Solution**: 
```bash
pip install opencv-python
```

### Inference Issues

**Problem**: "No face detected in image"  
**Solution**: 
- Don't use `--detect_face` flag (it will use full image)
- Or ensure the image actually contains a visible face

**Problem**: Low confidence predictions (<50%)  
**Solution**: 
- Ensure good lighting in the photo
- Use `--detect_face` to crop to just the face
- Check that the face is front-facing and clearly visible

**Problem**: Wrong predictions  
**Solution**: 
- Try ResNet50 (most accurate model)
- Use `--detect_face` to focus on facial region
- Ensure the expression is clear and not ambiguous

---

## Model Metadata

Each model comes with a JSON metadata file containing:
- **Preprocessing parameters** (mean, std, input size)
- **Class labels** (emotion names in correct order)
- **Training metrics** (validation/test accuracy)
- **Export information** (ONNX opset, PyTorch version)

Example (`resnet50_metadata.json`):
```json
{
  "model_name": "resnet50",
  "num_classes": 7,
  "class_labels": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"],
  "input_shape": [1, 3, 224, 224],
  "preprocessing": {
    "resize": [224, 224],
    "normalize": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },
  "training_info": {
    "final_test_acc": 0.744
  }
}
```

---

## FAQ

**Q: Do I need PyTorch installed?**  
A: No! ONNX Runtime is lightweight and doesn't require PyTorch.

**Q: Which model should I use?**  
A: ResNet50 for best accuracy (74.4%), MobileNet for fastest speed.

**Q: Can I use this on a Raspberry Pi?**  
A: Yes! Use the ONNX models with ONNX Runtime. MobileNet recommended for speed.

**Q: Does it work with webcam?**  
A: Use the full project's `webcam_demo.py`. The lightweight script is for static images.

**Q: How accurate are these models?**  
A: ResNet50 achieves 74.4% on our test set. Real-world accuracy varies with image quality.

**Q: Can I deploy this to a mobile app?**  
A: Yes! Convert ONNX to CoreML (iOS) or use ONNX Runtime Mobile (Android).

---

## Citation

If you use these models in your project, please reference:

```
Facial Emotion Detection System
Deep Learning Course Project
Using Transfer Learning with ResNet50/18, EfficientNet-B0, MobileNetV2
Trained on 3,198 facial expression images (7 emotion classes)
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify you have the correct dependencies installed
3. Ensure your images meet the input requirements
4. Contact the project team

---

## License

These models are provided for educational and research purposes.

---

**Happy Emotion Detection! 😊**
