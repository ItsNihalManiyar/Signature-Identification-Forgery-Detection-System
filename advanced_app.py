import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import tensorflow as tf
import time

# --- Layout & Theme ---
st.set_page_config(
    page_title="College Signature Auth",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Methods ---

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    id_model_path = os.path.join(base_dir, 'student_id_model.h5')
    verify_model_path = os.path.join(base_dir, 'forgery_detect_model.h5')
    
    id_model, verify_model = None, None
    if os.path.exists(id_model_path):
        id_model = tf.keras.models.load_model(id_model_path)
    if os.path.exists(verify_model_path):
        verify_model = tf.keras.models.load_model(verify_model_path)
    return id_model, verify_model

def get_student_names():
    NAMES = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna", "Ishaan", "Shaurya",
        "Atharv", "Advik", "Pranav", "Reyansh", "Moksh", "Dhruv", "Kabir", "Ritvik", "Aarush", "Kian",
        "Darsh", "Veer", "Ansh", "Laksh", "Dev", "Ishan", "Yash", "Samar", "Karaan", "Rudra",
        "Ananya", "Diya", "Saanvi", "Aditi", "Ira", "Kavya", "Ahana", "Riya", "Myra", "Navya",
        "Sara", "Meera", "Aarohi", "Priya", "Neha", "Pooja", "Anjali", "Shruti", "Swati", "Nidhi",
        "Roshni", "Simran", "Tanvi", "Vidya", "Kriti"
    ]
    return {i: name for i, name in enumerate(NAMES, start=0)}

def preprocess_image_for_model(img_pil):
    img = np.array(img_pil)
    if len(img.shape) == 3:
        img = img[:, :, ::-1]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    resized = cv2.resize(thresh, (224, 224))
    
    final_img = resized / 255.0
    final_img = final_img.reshape(1, 224, 224, 1)
    
    return final_img, gray, thresh

# --- Session State Management ---
if 'uploaded_file_id' not in st.session_state:
    st.session_state['uploaded_file_id'] = None

# --- Main Application ---
def main():
    
    # --- Sidebar Navigation ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1055/1055661.png", width=60)
    st.sidebar.title("Navigation")
    nav_option = st.sidebar.radio("Go to", ["Upload & Analyze", "Model Evaluation", "About"])

    id_model, verify_model = load_models()

    if nav_option == "About":
        st.title("About the System")
        st.write("""
        ### College Signature Identification & Forgery Detection System
        
        This advanced biometric authentication system is designed to secure administrative and exam processes in modern educational institutions.
        
        **Dual-Model Architecture:**
        - **ID_Model:** A Deep Convolutional Neural Network (CNN) trained with categorical cross-entropy to accurately identify signatures belonging to 55 registered students.
        - **Verify_Model:** A separate binary classification CNN utilizing a sigmoid output layer to probabilistically determine if the submitted signature is a genuine artifact or an unauthorized forgery.
        
        **Dataset:** 
        Trained on the highly challenging CEDAR signature dataset comprising exactly 24 original and 24 forged signatures for each individual writer.

        **Image Processing Pipeline:**
        Utilizes robust OpenCV pipelines mapping RGB inputs to grayscale and binarizing dynamic ink strokes intelligently using Otsu's optimal thresholding before neural processing.
        """)

    elif nav_option == "Model Evaluation":
        st.title("Model Evaluation Metrics")
        st.write("Performance evaluation of the dual Convolutional Neural Networks on the CEDAR dataset validation split.")
        
        st.subheader("ID_Model (55-Class Identification)")
        col1, col2 = st.columns(2)
        col1.metric("Validation Accuracy", "81.63%")
        col2.metric("Validation Loss", "0.7454")
        
        st.subheader("Verify_Model (Binary Forgery Detection)")
        col3, col4 = st.columns(2)
        col3.metric("Validation Accuracy", "80.49%")
        col4.metric("Validation Loss", "0.4619")
        
        st.info("Metrics reflect the final validation epoch during the training phase.")

    elif nav_option == "Upload & Analyze":
        # --- Main Interface Setup ---
        st.title("College Signature Identification & Forgery Detection System")
        
        if id_model is None or verify_model is None:
            st.error("Error: Trained models not found. Please train models first using `train_models.py`.")
            st.stop()

        # Session State logic for clearing
        def clear_upload():
            if 'uploader' in st.session_state:
                del st.session_state.uploader
        
        st.sidebar.markdown("---")
        st.sidebar.button("Clear Current Session", on_click=clear_upload, type="secondary")

        uploaded_file = st.file_uploader(
            "Drag and drop a signature image here", 
            type=["png", "jpg", "jpeg"], 
            key="uploader",
            help="Supported formats: PNG, JPG, JPEG. Optimal resolution >= 224x224."
        )

        if uploaded_file is not None:
            
            # --- Two Column Layout ---
            col_img, col_results = st.columns([1, 1], gap="large")

            image = Image.open(uploaded_file)
            
            with col_img:
                st.subheader("Uploaded Signature")
                st.image(image, use_container_width=True, caption="Source Image")

            with col_results:
                st.subheader("Analysis Settings")
                analyze_btn = st.button("Analyze Signature", type="primary", use_container_width=True)
                
                if analyze_btn:
                    # --- Real-time Processing Feedback ---
                    progress_text = "Running Computer Vision Pipeline..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    time.sleep(0.5)
                    my_bar.progress(30, text="Preprocessing: Grayscale & Otsu Thresholding...")
                    processed_img, gray_img, thresh_img = preprocess_image_for_model(image)
                    
                    time.sleep(0.5)
                    my_bar.progress(60, text="Executing CNN Forward Pass (ID & Verification)...")
                    
                    # Predictions
                    id_pred = id_model.predict(processed_img)
                    verify_pred = verify_model.predict(processed_img)
                    
                    time.sleep(0.5)
                    my_bar.progress(100, text="Analysis Complete!")
                    time.sleep(0.2)
                    my_bar.empty()
                    
                    predicted_class = np.argmax(id_pred, axis=1)[0]
                    student_name = get_student_names().get(predicted_class, "Unknown")
                    
                    # Verify_Model output: 0->Genuine, 1->Forged
                    forg_prob = float(verify_pred[0][0])
                    is_forged = forg_prob > 0.5
                    
                    st.markdown("---")
                    
                    # --- Results Display ---
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color:#4B5563; margin-bottom:5px;">Identified Student</h4>
                        <h2 style="color:#2563EB; margin-top:0px;">{student_name}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Status Validation
                    if not is_forged:
                        st.success("✅ **Status Validation: GENUINE SIGNATURE**", icon="🛡️")
                        confidence = (1.0 - forg_prob) * 100
                    else:
                        st.error("❌ **Status Validation: FORGERY DETECTED**", icon="🚨")
                        confidence = forg_prob * 100
                    
                    # Confidence Scores
                    st.write("**AI Confidence Score:**")
                    st.progress(int(confidence))
                    st.caption(f"Model is {confidence:.2f}% confident in this assessment.")
            
            # Show intermediate prep images if analyzed
            if 'thresh_img' in locals():
                st.markdown("---")
                st.subheader("Pipeline Telemetry")
                tab1, tab2 = st.tabs(["Binarized Matrix (Otsu)", "Grayscale Vector"])
                with tab1:
                    st.image(thresh_img, caption="Cleaned Signature Trace via Otsu's Bimodal Thresholding", use_container_width=True)
                with tab2:
                    st.image(gray_img, caption="Flattened Grayscale Channel", use_container_width=True)


if __name__ == "__main__":
    main()
