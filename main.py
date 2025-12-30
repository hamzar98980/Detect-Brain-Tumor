import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
import requests
import os

# Load your trained model
# model = tf.keras.models.load_model("fine_tuning_with_resnet.keras")

# MODEL_URL = "https://huggingface.co/albereinsten/Brain-Tumor/blob/main/fine_tuning_with_resnet.keras"
MODEL_URL = "https://huggingface.co/albereinsten/Brain-Tumor/resolve/main/fine_tuning_with_resnet.keras"
MODEL_PATH = "fine_tuning_with_resnet.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()  # <-- IMPORTANT
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

download_model()

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image and the model will predict whether a tumor is present.")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # --- PREPROCESSING (Correct method) ---
    img = img.resize((180, 180))                # Resize to model input
    img_array = np.array(img)                   # Convert to array (180,180,3)
    img_array = np.expand_dims(img_array, 0)    # Convert to (1,180,180,3)
    img_array = preprocess_input(img_array)     # Apply ResNet preprocessing

    # --- PREDICT ---
    prediction = model.predict(img_array)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor"

    # --- UI OUTPUT ---
    st.subheader("Prediction Result")
    st.write(f"**Model Output:** {prediction:.4f}")

    if prediction > 0.5:
        st.error("🔴 Tumor Detected")
    else:
        st.success("🟢 No Tumor Detected")
