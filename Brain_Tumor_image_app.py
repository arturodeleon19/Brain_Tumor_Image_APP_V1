import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# 2. Load the Model (Cached so it only loads once)
@st.cache_resource
def load_mri_model():
    # Ensure the model file is in the same directory as this script
    model_path = 'brain_tumor_EfficientB0_FT.keras'
    return tf.keras.models.load_model(model_path)

try:
    model = load_mri_model()
except Exception as e:
    st.error(f"Error loading model. Please ensure 'brain_tumor_EfficientB0_FT.keras' is in the directory. Details: {e}")
    st.stop()

# 3. Prediction Function
def predict_tumor(image):
    # Convert to RGB and resize to match model input (224x224)
    img = image.convert('RGB')
    img = img.resize((224, 224))
    
    # Convert to numpy array and add batch dimension (1, 224, 224, 3)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: EfficientNet handles its own [0, 255] scaling internally, so no division by 255.0
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    # Sigmoid output: > 0.5 is Tumor (Class 1), <= 0.5 is No Tumor (Class 0)
    if probability > 0.5:
        return "Tumor Detected", probability
    else:
        # For No Tumor, invert the probability so the UI shows confidence correctly
        return "No Tumor", (1 - probability)

# 4. App UI
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan or select one of the pre-loaded validation samples to detect the presence of a tumor.")

# Create tabs for clean organization
tab1, tab2 = st.tabs(["📁 Try a Sample Image", "⬆️ Upload Your Own"])

# --- TAB 1: Sample Images ---
with tab1:
    st.subheader("Test with Pre-loaded Images")
    sample_dir = "sample_images"
    
    if os.path.exists(sample_dir) and os.listdir(sample_dir):
        # Get list of images in the sample_images folder
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_files:
            selected_sample = st.selectbox("Choose a sample MRI:", sample_files)
            sample_path = os.path.join(sample_dir, selected_sample)
            
            # Display the selected image
            sample_image = Image.open(sample_path)
            st.image(sample_image, caption=f"Selected: {selected_sample}", width=350)
            
            if st.button("Analyze Sample"):
                with st.spinner("Analyzing scan..."):
                    label, confidence = predict_tumor(sample_image)
                    
                    if label == "Tumor Detected":
                        st.error(f"**Result:** {label} (Confidence: {confidence:.2%})")
                    else:
                        st.success(f"**Result:** {label} (Confidence: {confidence:.2%})")
        else:
            st.info("No images found in the 'sample_images' folder. Please add some .jpg or .png files.")
    else:
        st.warning("The 'sample_images' folder does not exist. Please create it and add validation images to test this feature.")

# --- TAB 2: File Upload ---
with tab2:
    st.subheader("Upload an MRI Scan")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="Uploaded MRI Scan", width=350)
        
        if st.button("Analyze Upload"):
            with st.spinner("Analyzing scan..."):
                label, confidence = predict_tumor(user_image)
                
                if label == "Tumor Detected":
                    st.error(f"**Result:** {label} (Confidence: {confidence:.2%})")
                else:
                    st.success(f"**Result:** {label} (Confidence: {confidence:.2%})")
