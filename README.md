# MRI Tumor Classification with Deep Learning

This repository contains a deep learning pipeline that classifies brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. The project leverages a pre-trained **VGG16** model with transfer learning and visual explanation techniques like **Grad-CAM** and **Occlusion Sensitivity Maps** for interpretable predictions.

---

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and includes labeled MRI scans organized by tumor type. It is automatically downloaded via `kagglehub`.

---

## Preprocessing and Its Role in Tumor Detection

Before training, MRI images go through several preprocessing steps that prepare the data and improve model performance:

###  Steps:

1. **Resizing**: All images are resized to **128x128 pixels** for consistency and efficient processing.

2. **Data Augmentation**:
   - Brightness, contrast, and color adjustments
   - Introduces variation to prevent overfitting and simulate real-world noise

3. **Normalization**: Pixel values are scaled to a [0, 1] range to help with faster and more stable training.

4. **Label Encoding**: Tumor classes are mapped to numeric values for categorical classification.

###  Why It Matters:
These steps enhance the model's ability to detect subtle patterns in MRI scans, such as irregular textures or shapes associated with tumors, improving its accuracy and generalization.

---

##  Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Modifications**:
  - Last 3 convolutional layers fine-tuned
  - Added dense layers with dropout
- **Loss**: Sparse Categorical Crossentropy
- **Optimizer**: Adam (`lr = 0.0001`)
- **Training**: 6 epochs with batch size 20

---

## 📊 Evaluation

- Confusion matrix
- Accuracy and loss plots
- Classification report

---

##  Model Interpretability

###  Grad-CAM:
Visualizes areas the model focuses on while making predictions.

###  Occlusion Maps:
Reveals sensitive regions by occluding parts of the image and measuring prediction drop.

These tools provide transparency, especially in medical diagnostics.

---

## Sample Usage

# Predict and display result on a test image
detect_and_display("path_to_image.jpg", model)

# Visualize Grad-CAM
analyze_samples_with_gradcam(test_paths, model, class_labels, num_samples=5)
