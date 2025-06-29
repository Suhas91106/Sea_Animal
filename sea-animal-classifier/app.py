import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set up Streamlit page
st.set_page_config(page_title="Sea Animal Classifier", layout="centered")

# Load model
@st.cache_resource
def load_trained_model():
    model = load_model("DenseNet201_model80.h5")  # Ensure this is in your repo
    return model

model = load_trained_model()

# Define class labels
class_labels = [
    'Dolphin', 'Fish', 'Jelly Fish', 'Octopus', 'Penguin', 'Sea Rays',
    'Sea Urchins', 'Seahorse', 'Sharks', 'Starfish', 'Turtle_Tortoise',
    'Whale', 'crab'
]

# Prediction function
def predict_image(uploaded_file, model, class_labels):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((250, 250))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction)
    predicted_label = class_labels[predicted_idx]
    confidence = float(np.max(prediction))

    return predicted_label, confidence, img

# UI layout
st.title("🐠 Sea Animal Image Classifier")
st.markdown("Upload an image of a sea animal. The model will predict what it is!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        label, confidence, image_display = predict_image(uploaded_file, model, class_labels)

    st.success(f"✅ Prediction: **{label}**")
    st.info(f"🔍 Confidence: **{confidence*100:.2f}%**")
    st.image(image_display, caption=f"Predicted: {label}", use_column_width=True)

st.markdown("---")
st.caption("Powered by DenseNet201 & Streamlit 💙")
