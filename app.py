import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Class names
class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in-scale', 'scratches']

# Load model
model = tf.keras.models.load_model("neu_defect_deploy.keras")

# Streamlit UI
st.title("🔍 Metal Surface Defect Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    img = img.resize((128,128))
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    st.success(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: {confidence:.2f}%")