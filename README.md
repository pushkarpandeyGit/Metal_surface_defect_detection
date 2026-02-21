# Metal Surface Defect Detection

## Problem Statement

In metal manufacturing, surfaces often develop defects such as scratches, inclusions, pitting, rolled-in scale, patches, and crazing. Manual inspection to detect these defects is time-consuming, labor-intensive, and prone to human error. Early detection of these defects is crucial for maintaining product quality and reducing manufacturing costs.

The challenge is to automate the process of detecting different types of defects on metal surfaces efficiently and accurately.

---

## Project Solution

This project provides an automated solution for detecting metal surface defects using deep learning. The solution includes:

1. **A Convolutional Neural Network (CNN) model** based on MobileNetV2 that is trained to classify images into six types of surface defects.
2. **Data augmentation** techniques to improve model generalization and reduce overfitting.
3. **Streamlit web application** for end-users to upload metal surface images and receive real-time predictions of defect type with confidence scores.

With this system, quality control teams can quickly and accurately identify defects, reducing reliance on manual inspection.

---

## Dataset

The model is trained and validated on the **NEU Surface Defect Dataset**. The dataset contains images of metal surfaces with six types of defects:

- Crazing  
- Inclusion  
- Patches  
- Pitted Surface  
- Rolled-in Scale  
- Scratches  

Training and validation images are resized to **128x128 pixels** for input to the model. A batch size of 32 is used during training. Data augmentation techniques applied include:

- Random horizontal flipping  
- Random rotation  
- Random zoom  
- Random contrast adjustments  
- Random translation  

These augmentations help improve the model's ability to generalize to unseen images.

---

## Technical Approach

1. **Model Architecture:**
   - Base model: MobileNetV2 pretrained on ImageNet (frozen during training)  
   - Custom layers added:
     - Data augmentation  
     - Lambda layer for preprocessing  
     - GlobalAveragePooling2D  
     - Dropout layer for regularization  
     - Dense output layer with 6 neurons and softmax activation  

2. **Training:**
   - Optimizer: Adam  
   - Loss function: Sparse Categorical Crossentropy  
   - Metrics: Accuracy  
   - Early stopping applied to prevent overfitting  

3. **Deployment:**
   - The trained model is saved as `neu_defect_deploy.keras`  
   - A Streamlit app (`app.py`) allows users to upload images and get defect predictions in real-time.  

4. **Streamlit Usage:**
   - Streamlit is used to create a simple web interface without the need for HTML, CSS, or JavaScript.  
   - Users can upload an image, which is then resized and processed to match the model input.  
   - The app displays the uploaded image, predicts the defect type, and shows the confidence percentage.

---

## Libraries Used

- **TensorFlow / Keras** – For building and training the deep learning model  
- **NumPy** – For array operations and image preprocessing  
- **Pillow (PIL)** – For handling image input and conversions  
- **Streamlit** – For creating the interactive web application  
- **OS / Python standard libraries** – For file handling  

---

## Running the Project

1. Install dependencies:


pip install tensorflow streamlit pillow numpy
streamlit run app.py

<img width="1259" height="909" alt="image" src="https://github.com/user-attachments/assets/eb97d53c-e600-4328-bab1-047bf87ad625" />
<img width="1268" height="912" alt="Screenshot 2026-02-20 151849" src="https://github.com/user-attachments/assets/e744e9c2-91f3-47e6-a025-b68a948cace4" />

## Deployment link
https://metalsurfacedefectdetection-ddyxmrhhr88mcjbui8cfyu.streamlit.app/

## Future Improvements

Increase dataset size to improve model accuracy.

Experiment with different architectures such as EfficientNet or ResNet for potentially higher performance.

Add real-time camera input in the Streamlit app for live defect detection.

Integrate the system into a manufacturing pipeline for automatic quality control.

