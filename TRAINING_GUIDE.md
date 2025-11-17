# Expression Detection - Fine-Tuning Guide

## Overview
This guide explains how to fine-tune the expression detection model on your dataset to improve accuracy.

## Dataset Structure

Organize your dataset in the following folder structure:

```
dataset/
├── happy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── sad/
│   ├── image1.jpg
│   └── ...
├── angry/
├── surprised/
├── neutral/
├── fear/
└── disgust/
```

## Training the Model
Note:codebase also supports (with the same interface):

ResNet50: 'resnet50'

MobileNetV3: 'mobilenet_v3'

But unless you provide --model resnet50 or similar at the command line, the default remains EfficientNet-B0.

### Basic Training
```bash
python expression-detection-optimized.py --mode train --data_dir "dataset/train" --epochs 10 --batch_size 32  
```
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
### Parameters:
- `--mode`: Set to `train` for training
- `--data_dir`: Path to your dataset folder
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)


## Using FER2013 Dataset

If you have FER2013 dataset, organize it as:
```
fer2013/
├── Angry/
├── Disgust/
├── Fear/
├── Happy/
├── Sad/
├── Surprise/
└── Neutral/
```

Then modify the `model` call to use `is_fer2013=True` in the ExpressionDataset.




## Model Checkpoint

The best model is automatically saved to `best_expression_model_checkpoint.pth` during training. The model will automatically load this checkpoint for inference if it exists.

## Running Inference After Training

```bash
# Process images from input folder


python expression-detection-optimized.py --mode inference --model_path models/best_model.pth --input_folder snapshots --output_folder inference_results

# Process video feed
python experience_detection.py --mode video
```

## for Better Accuracy

1. **More Data**: Use at least 100-200 images per expression class
2. **Balanced Dataset**: Ensure roughly equal samples per class
3. **Data Quality**: Use clear, well-lit face images
4. **More Epochs**: Train for 20-30 epochs for better convergence
5. **Larger Model**: Try `--model_name resnet50` for better accuracy (slower)

## Troubleshooting

- **No images found**: Check your dataset folder structure
- **CUDA out of memory**: Reduce `--batch_size` (e.g., 16 or 8)
- **Low accuracy**: Increase epochs, check data quality, add more training data

