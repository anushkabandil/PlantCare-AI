import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("../model/plant_model.h5")

# Class labels
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

# Disease descriptions
disease_info = {
    "Tomato Early Blight": "A fungal disease causing brown spots on leaves.",
    "Tomato Late Blight": "A serious disease causing large dark lesions on leaves and fruit.",
    "Tomato Healthy": "The plant appears healthy with no visible disease symptoms."
}

treatment_info = {
    "Tomato Early Blight": "Remove infected leaves and apply fungicide.",
    "Tomato Late Blight": "Use copper-based fungicide and remove infected plants.",
    "Tomato Healthy": "No treatment required. Maintain proper watering and sunlight."
}

# Page config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect possible diseases.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "jfif"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Leaf")

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    disease = class_names[predicted_class]

    with col2:
        st.subheader("Prediction Result")
        st.success(disease)

        st.write("Confidence")
        st.progress(float(confidence))

        st.write(f"{confidence*100:.2f}%")

        st.subheader("About the Disease")
        st.info(disease_info[disease])

        st.subheader("Treatment Advice")
        st.warning(treatment_info[disease])