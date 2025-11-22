### Project Tasks for Facial Emotion Detection

#### **Phase 1: Data Preparation**

1. **Data Cleaning**:
    * Remove corrupted, blurry, or non-face images.
    * Ensure consistent file formats (e.g., convert all to JPG/PNG).
2. **Preprocessing**:
    * **Face Cropping**: Use a face detector (like Haar Cascades or MTCNN) to crop just the face from each image, removing background noise.
    * **Resizing**: Resize all images to **224x224 pixels** (required input size for MobileNetV2/ResNet).
    * **Normalization**: Scale pixel values to the  or [-1, 1] range.[^1]
3. **Data Splitting**:
    * Split your ~3,300 images into **Training (80%)**, **Validation (10%)**, and **Test (10%)** sets. Ensure each emotion class is balanced across splits.
4. **Data Augmentation** (Crucial for small data):
    * Apply random transformations to training data: rotation (±10°), width/height shifts, horizontal flips, and brightness adjustments. This artificially expands your dataset size and prevents overfitting.

#### **Phase 2: Model Implementation**

5. **Base Model Setup**:
    * Load a pretrained model (e.g., **MobileNetV2** or **ResNet18**) with weights from ImageNet.
    * **Freeze** the base layers so their weights don't update during the initial training.
6. **Custom Head**:
    * Add a new top layer structure: `GlobalAveragePooling2D` -> `Dense (128, ReLU)` -> `Dropout (0.5)` -> `Dense (num_classes, Softmax)`.
    * The dropout layer helps further reduce overfitting.

#### **Phase 3: Training**

7. **Compile Model**:
    * Use **Categorical Crossentropy** as the loss function.
    * Use **Adam** optimizer with a learning rate of roughly `1e-3`.
8. **Initial Training (Warm-up)**:
    * Train for 10-20 epochs with frozen base layers.
    * Monitor validation accuracy/loss.
9. **Fine-Tuning**:
    * **Unfreeze** the last few blocks of the base model.
    * Re-compile with a much lower learning rate (e.g., `1e-5`).
    * Train for another 10-20 epochs. Use **EarlyStopping** callback to stop if validation loss doesn't improve for 3-5 epochs.

#### **Phase 4: Evaluation \& Deployment**

10. **Evaluation**:
    * Run inference on the **Test Set**.
    * Generate a **Confusion Matrix** to see which emotions are confused with each other (e.g., "Fear" vs. "Surprise").
11. **Inference Pipeline**:
    * Create a script that captures video (webcam), detects faces, applies the same preprocessing (crop + resize + normalize), and feeds it to the model for real-time prediction.
12. **Optimization (Optional)**:
    * Convert the trained model to **ONNX** or **TFLite** format for faster inference on Windows/Edge devices.

This pipeline directly addresses the limitations of a small dataset by leveraging powerful pretrained features and rigorous augmentation.