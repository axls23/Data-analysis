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

## Next Steps

Phase 2: Model Training (Coming soon...)

## Project Structure

```
edl-project_facial-emotion-detection/
├── dataset/                       # Raw dataset
├── preprocessed_data/             # Preprocessed images (Phase 1)
├── data_splits/                   # Train/val/test splits (Phase 1)
├── scripts/                       # PowerShell utility scripts
│   ├── clean_mac_files.ps1
│   ├── validate_naming.ps1
│   ├── organize_dataset.ps1
│   └── process_zip_datasets.ps1
├── preprocess_dataset.py          # Image preprocessing pipeline
├── split_dataset.py               # Dataset splitting pipeline
├── expression-detection-optimized.py  # Training & inference (Phase 2+)
├── PROJECT_TASKS.md               # Detailed project tasks
├── TRAINING_GUIDE.md              # Training documentation
└── README.md                      # This file
```
