import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    """Load the trained lung disease classification model."""
    model_path = "lung_disease_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please upload 'lung_disease_model.h5' in the project directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# Prediction function
def predict(image_file, model):
    """Process the image and make predictions using the model."""
    if model is None:
        return None
    
    try:
        img = Image.open(image_file).convert('RGB').resize((224, 224))  # Convert to RGB & resize
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Debugging: Print image shape
        st.write(f"🔍 Processed Image Shape: {img_array.shape}")
        
        predictions = model.predict(img_array)
        class_names = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]  # Modify based on dataset
        results = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        
        return results
    except ValueError as e:
        st.error(f"❌ Model Input Error: {str(e)}")
        return None

# Main Streamlit App
def main():
    """Streamlit UI for Lung Disease Classification."""
    st.set_page_config(page_title="Lung Disease Classifier", page_icon="🫁", layout="centered")
    
    st.title("🫁 Lung Disease Classifier")
    st.write("Upload a chest X-ray image, and the AI will classify it!")

    with st.sidebar:
        st.header("🔍 About the Model")
        st.write("This model detects lung diseases such as **Pneumonia, Tuberculosis, and COVID-19** from chest X-ray images.")
        st.write("Built using **Deep Learning** and **TensorFlow/Keras**.")
        st.write("📌 **Instructions:**")
        st.write("1️⃣ Upload a **chest X-ray image** (JPG, PNG, JPEG).")
        st.write("2️⃣ The model will analyze and classify the image.")
        st.write("3️⃣ The predicted probability for each class will be displayed.")

    model = load_model()

    uploaded_file = st.file_uploader("📂 Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 Uploaded X-ray", use_column_width=True)
        st.write("🧐 **Analyzing... Please wait.**")

        predictions = predict(uploaded_file, model)

        if predictions:
            st.subheader("📊 **Classification Results:**")
            for disease, score in predictions.items():
                st.write(f"🔹 **{disease}**: {score * 100:.2f}%")

            # Highlight the most likely condition
            top_disease = max(predictions, key=predictions.get)
            st.success(f"✅ **Most likely diagnosis:** {top_disease} ({predictions[top_disease] * 100:.2f}%)")
        else:
            st.error("❌ Prediction failed. Please check if the model is correctly loaded.")

if __name__ == "__main__":
    main()
