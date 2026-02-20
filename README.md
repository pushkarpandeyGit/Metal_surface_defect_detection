# 🔍 Metal Surface Defect Detection

This project is a **metal surface defect detection application** built using **TensorFlow** and deployed with **Streamlit**. It detects six types of surface defects on metal sheets using a **MobileNetV2-based CNN**.

---

## 🛠️ Features

- Detects 6 types of metal surface defects:
  1. Crazing
  2. Inclusion
  3. Patches
  4. Pitted Surface
  5. Rolled-in Scale
  6. Scratches
- Uses **MobileNetV2** as the base model with data augmentation to improve accuracy.
- Streamlit-based web interface for easy image upload and defect prediction.
- Displays prediction with confidence percentage.

---

## 📂 Dataset

- **Training images:** `neu_surface_defect/NEU-DET/train/images`
- **Validation images:** `neu_surface_defect/NEU-DET/validation/images`
- Images are preprocessed to size **128x128 pixels**.
- **Batch size:** 32
- **Data augmentation:** Horizontal flip, random rotation, zoom, contrast, and translation to reduce overfitting.

---

## 🧠 Model

- Base model: **MobileNetV2** (pretrained on ImageNet, frozen)
- Additional layers:
  - Data augmentation
  - GlobalAveragePooling2D
  - Dropout (0.5)
  - Dense output layer with 6 neurons and softmax activation
- **Loss function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Early stopping:** Patience = 5 epochs to reduce overfitting

```python
# Example of model summary
model.summary()
