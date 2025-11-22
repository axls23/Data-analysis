# Facial Emotion Detection System

A deep learning-based facial emotion detection system using transfer learning with pretrained models (EfficientNet-B0, ResNet50, MobileNetV3).

## Project Overview

This project implements a comprehensive pipeline for detecting and classifying facial emotions into 7 categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprised

## Dataset Structure

The dataset follows a strict naming convention:
```
dataset/
├── d1/                    # Dataset 1
│   ├── angry/
│   │   ├── 23XXX-01-AN-01.jpg
│   │   ├── 23XXX-01-AN-02.jpg
│   │   └── ...
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
├── d2/                    # Dataset 2
└── ...                    # Additional datasets
```

### Naming Convention
All images follow the format: `<USN>-<PersonNumber>-<EmotionTag>-<ImageNumber>.jpg`

- **USN**: University Serial Number starting with "23" (e.g., `23BTRCL202`)
- **PersonNumber**: `01`, `02`, or `03`
- **EmotionTag**: Two-letter emotion code:
  - `AN` - Angry
  - `DI` - Disgust
  - `FE` - Fear
  - `HA` - Happy
  - `NE` - Neutral
  - `SA` - Sad
  - `SU` - Surprised
- **ImageNumber**: `01`, `02`, or `03`

**Example**: `23BTRCL202-01-HA-01.jpg` (Person 01, Happy emotion, Image 01)

## Phase 1: Data Preparation ✅

### Prerequisites

```bash
pip install torch torchvision opencv-python pillow numpy tqdm
# Optional for better face detection:
pip install facenet-pytorch
```

### Step 1: Dataset Organization and Validation

#### 1.1 Clean macOS Metadata Files
```powershell
.\scripts\clean_mac_files.ps1
```
Removes `._ ` files created by macOS that can interfere with processing.

#### 1.2 Validate and Fix Naming Conventions
```powershell
# Dry run to see what would be changed
.\scripts\validate_naming.ps1 -DryRun

# Apply fixes
.\scripts\validate_naming.ps1
```

Features:
- Converts underscores to hyphens
- Adds zero-padding to numbers
- Normalizes emotion codes
- Removes spaces and extra formatting
- Enforces `.jpg` extension
- Handles 'O' vs '0' typos
- Smart emotion tag inference

#### 1.3 Organize Images into Emotion Folders
```powershell
.\scripts\organize_dataset.ps1
```

Moves images from dataset roots into their respective emotion subfolders.

### Step 2: Image Preprocessing

Preprocesses all images with face detection, cropping, and resizing to 224×224 pixels.

```bash
# Using Haar Cascade (default, no extra dependencies)
python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data

# Using MTCNN (better accuracy, requires facenet-pytorch)
python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data --detector mtcnn

# Custom settings
python preprocess_dataset.py --input_dir dataset --output_dir preprocessed_data --margin 0.3 --size 256 --skip_existing
```

**Options:**
- `--detector`: `haar` (default) or `mtcnn`
- `--margin`: Margin around detected face (default: 0.2)
- `--size`: Target image size (default: 224)
- `--skip_existing`: Skip already processed images

**Output:**
- Preprocessed images in `preprocessed_data/` with same structure as input
- `preprocessing_stats.json` with detailed statistics

### Step 3: Create Train/Validation/Test Splits

Creates stratified splits while maintaining balanced emotion class distribution.

```bash
# Default 80/10/10 split
python split_dataset.py --input_dir preprocessed_data --output_dir data_splits

# Custom split ratios (70/15/15)
python split_dataset.py --input_dir preprocessed_data --output_dir data_splits --train_ratio 0.7 --val_ratio 0.15

# Use different random seed
python split_dataset.py --input_dir preprocessed_data --output_dir data_splits --seed 123
```

**Output Structure:**
```
data_splits/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
└── split_info.json
```

---

## Phase 2: Model Architecture Implementation ✅

### Overview

Phase 2 implements a modular, scalable architecture supporting multiple state-of-the-art deep learning models for emotion recognition. All models leverage transfer learning with ImageNet pretrained weights and custom classification heads optimized for 7-class emotion detection.

### Implemented Models

We implemented four production-ready convolutional neural network architectures, each with distinct characteristics suited for different deployment scenarios:

#### 1. **MobileNetV2** (2.4M parameters)
- **Architecture**: Inverted residual structure with linear bottlenecks
- **Strengths**: Lightweight, optimized for mobile and edge devices
- **Use Case**: Real-time inference on resource-constrained devices
- **Speed**: Fastest inference time
- **Pretrained Weights**: ImageNet-1K (1000 classes)

#### 2. **EfficientNet-B0** (4.2M parameters)
- **Architecture**: Compound scaling method (depth + width + resolution)
- **Strengths**: State-of-the-art parameter efficiency
- **Use Case**: Best accuracy-to-parameter ratio
- **Speed**: Fast inference with excellent accuracy
- **Pretrained Weights**: ImageNet-1K (1000 classes)

#### 3. **ResNet18** (11.2M parameters)
- **Architecture**: Residual connections with skip connections
- **Strengths**: Strong baseline, stable training
- **Use Case**: General-purpose emotion recognition
- **Speed**: Moderate inference time
- **Pretrained Weights**: ImageNet-1K (1000 classes)

#### 4. **ResNet50** (23.8M parameters)
- **Architecture**: Deeper residual network with bottleneck blocks
- **Strengths**: Maximum feature extraction capability
- **Use Case**: Highest accuracy when computational resources available
- **Speed**: Slower but most accurate
- **Pretrained Weights**: ImageNet-1K (1000 classes)

### Model Architecture Design

All models follow a consistent two-stage architecture:

#### Stage 1: Feature Extraction (Frozen Backbone)
```
Input Image (224×224×3)
    ↓
Pretrained CNN Backbone (ImageNet weights)
    ↓
Feature Maps
```

#### Stage 2: Custom Classification Head
```
Feature Maps
    ↓
Linear(backbone_features → 128)
    ↓
ReLU Activation
    ↓
Dropout(p=0.5)
    ↓
Linear(128 → 7)
    ↓
Output Logits (7 emotion classes)
```

**Design Rationale:**
- **Hidden Layer (128 units)**: Provides sufficient representational capacity while preventing overfitting
- **ReLU Activation**: Non-linearity for complex emotion pattern learning
- **Dropout (50%)**: Regularization to combat overfitting on small dataset (~3,300 images)
- **Output Layer**: 7 units for emotion classes (CrossEntropyLoss handles softmax internally)

### Transfer Learning Strategy

Our implementation supports a two-phase training approach:

#### Phase 3a: Warm-up Training (Planned)
- **Frozen Backbone**: All pretrained layers frozen
- **Trainable**: Only custom classification head
- **Purpose**: Adapt classifier to emotion recognition task
- **Typical Duration**: 10-20 epochs
- **Learning Rate**: ~1×10⁻³

#### Phase 3b: Fine-Tuning (Planned)
- **Unfrozen Backbone**: Last N layers unfrozen
- **Trainable**: Classifier + top backbone layers
- **Purpose**: Refine features for facial emotion patterns
- **Typical Duration**: 10-20 epochs
- **Learning Rate**: ~1×10⁻⁵ (much lower)

### Code Organization

The implementation follows a modular, object-oriented design for maintainability and extensibility:

```
models/
├── __init__.py              # Package initialization and exports
├── base_model.py            # Abstract base class (BaseEmotionModel)
├── mobilenet.py             # MobileNetV2 implementation
├── resnet.py                # ResNet18 and ResNet50 implementations
├── efficientnet.py          # EfficientNet-B0 implementation
└── model_factory.py         # Factory pattern for model creation

config/
├── __init__.py              # Configuration exports
└── model_config.py          # Hyperparameters and training config
```

### Model Factory Pattern

Simplified model instantiation through factory pattern:

```python
from models import create_model

# Create models with default settings
model = create_model('mobilenet', num_classes=7, pretrained=True)
model = create_model('resnet18', num_classes=7, pretrained=True)
model = create_model('efficientnet', num_classes=7, pretrained=True)

# Custom hyperparameters
model = create_model('resnet50', 
                    num_classes=7, 
                    pretrained=True,
                    hidden_dim=256,    # Custom hidden layer size
                    dropout=0.3)       # Custom dropout rate
```

### Base Model Interface

All models inherit from `BaseEmotionModel` providing:

**Core Methods:**
- `forward(x)`: Forward pass through model
- `freeze_backbone()`: Freeze all pretrained layers
- `unfreeze_backbone(num_layers)`: Selectively unfreeze layers
- `get_trainable_params()`: Get parameter counts
- `print_summary()`: Display model statistics

**Example Usage:**
```python
model = create_model('mobilenet')

# Freeze backbone for warm-up training
model.freeze_backbone()
# Only classifier layers are trainable (6.9% of parameters)

# Later, unfreeze for fine-tuning
model.unfreeze_backbone()
# All layers trainable (100% of parameters)
```

### Model Testing and Validation

Comprehensive testing suite validates all models:

```bash
# Test all models
python test_models.py

# Test specific model
python test_models.py --model resnet18

# Quiet mode (minimal output)
python test_models.py --quiet
```

**Test Coverage:**
✅ Model instantiation with pretrained weights  
✅ Forward pass with dummy data (batch_size=4, 224×224×3)  
✅ Output shape validation (batch_size, 7)  
✅ Freeze/unfreeze functionality  
✅ Parameter counting accuracy  
✅ GPU detection and utilization  

### Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install numpy pillow opencv-python
pip install tqdm

# Optional for better face detection (Phase 1)
pip install facenet-pytorch
```

### Model Comparison Summary

| Model | Parameters | Frozen Params | Trainable (Head) | Best For | Speed Rank |
|-------|-----------|---------------|------------------|----------|------------|
| MobileNetV2 | 2.4M | 2.22M | 165K (6.9%) | Mobile/Edge | 🥇 Fastest |
| EfficientNet-B0 | 4.2M | 4.01M | 165K (4.0%) | Efficiency | 🥈 Fast |
| ResNet18 | 11.2M | 11.18M | 67K (0.6%) | General Use | 🥉 Moderate |
| ResNet50 | 23.8M | 23.51M | 263K (1.1%) | Max Accuracy | ⚠️ Slower |

### Configuration Management

Centralized configuration in `config/model_config.py`:

```python
# Model architecture
NUM_CLASSES = 7
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
INPUT_SIZE = 224
HIDDEN_DIM = 128
DROPOUT_RATE = 0.5

# Training hyperparameters (Phase 3)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3        # Warm-up phase
FINE_TUNE_LR = 1e-5         # Fine-tuning phase
NUM_EPOCHS_WARMUP = 20
NUM_EPOCHS_FINETUNE = 20
EARLY_STOPPING_PATIENCE = 5

# Data augmentation settings (Phase 3)
AUGMENTATION = {
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'brightness_range': (0.8, 1.2),
    'zoom_range': 0.1,
}
```

### Validation Results

All four models successfully passed validation:

```
======================================================================
MODEL COMPARISON
======================================================================
Model                Total Params    Status     Notes
----------------------------------------------------------------------
mobilenet            2,388,743       ✓ PASS
resnet18             11,243,079      ✓ PASS
resnet50             23,771,207      ✓ PASS
efficientnet         4,172,419       ✓ PASS
======================================================================

Summary: 4/4 models passed all tests
✓ All models are ready for Phase 3 training!
```

### Next Steps

**Phase 3: Model Training** (In Progress)
- Implement data augmentation pipeline
- Create training loop with early stopping
- Implement learning rate scheduling
- Train all 4 models with warm-up + fine-tuning
- Compare model performance metrics

**Phase 4: Evaluation & Deployment**
- Generate confusion matrices
- Analyze per-emotion accuracy
- Real-time webcam inference
- Model optimization (ONNX/TFLite conversion)

---

## Project Structure

```
edl-project_facial-emotion-detection/
├── dataset/                       # Raw dataset
├── preprocessed_data/             # Preprocessed images (Phase 1)
├── data_splits/                   # Train/val/test splits (Phase 1)
├── models/                        # Model architectures (Phase 2) ✅
│   ├── __init__.py
│   ├── base_model.py             # Abstract base class
│   ├── mobilenet.py              # MobileNetV2 implementation
│   ├── resnet.py                 # ResNet18/50 implementations
│   ├── efficientnet.py           # EfficientNet-B0 implementation
│   └── model_factory.py          # Factory pattern
├── config/                        # Configuration (Phase 2) ✅
│   ├── __init__.py
│   └── model_config.py           # Hyperparameters
├── scripts/                       # PowerShell utility scripts
│   ├── clean_mac_files.ps1
│   ├── validate_naming.ps1
│   ├── organize_dataset.ps1
│   └── process_zip_datasets.ps1
├── preprocess_dataset.py          # Image preprocessing pipeline
├── split_dataset.py               # Dataset splitting pipeline
├── test_models.py                 # Model validation script (Phase 2) ✅
├── expression-detection-optimized.py  # Training & inference (Phase 2+)
├── PROJECT_TASKS.md               # Detailed project tasks
├── TRAINING_GUIDE.md              # Training documentation
└── README.md                      # This file
```
