import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf

@st.cache_resource
def load_models():
    # Load models if they exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    id_model_path = os.path.join(base_dir, 'student_id_model.h5')
    verify_model_path = os.path.join(base_dir, 'forgery_detect_model.h5')
    
    id_model = None
    verify_model = None
    
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
    # Model maps 1-55 to 0-54
    return {i: name for i, name in enumerate(NAMES, start=0)}

def preprocess_image_for_model(img_pil):
    # Convert PIL directly to OpenCV format
    img = np.array(img_pil)
    # Convert RGB to BGR
    if len(img.shape) == 3:
        img = img[:, :, ::-1]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Resize
    resized = cv2.resize(thresh, (224, 224))
    
    # Normalize and reshape
    final_img = resized / 255.0
    final_img = final_img.reshape(1, 224, 224, 1)
    
    return final_img

def main():
    st.set_page_config(page_title="Signature Identification & Forgery Detection", layout="centered")
    st.title("College Signature Identification & Forgery Detection System")

    id_model, verify_model = load_models()
    if id_model is None or verify_model is None:
        st.warning("Models not found! Please train the models by running train_models.py first.")

    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Upload Signature", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Signature', width='stretch')

        if st.button('Analyze Signature'):
            if id_model is not None and verify_model is not None:
                with st.spinner("Analyzing..."):
                    processed_img = preprocess_image_for_model(image)
                    
                    # Predictions
                    id_pred = id_model.predict(processed_img)
                    verify_pred = verify_model.predict(processed_img)
                    
                    # Extract results
                    predicted_class = np.argmax(id_pred, axis=1)[0]
                    id_to_name = get_student_names()
                    student_name = id_to_name.get(predicted_class, "Unknown")
                    
                    # Verify_Model output: closer to 0 is Genuine, closer to 1 is Forged
                    is_forged = int(verify_pred[0][0] > 0.5)
                    
                st.subheader("Results")
                st.write(f"**Identified Student:** {student_name}")
                
                if is_forged == 0:
                    st.success("Status: Genuine")
                else:
                    st.error("Status: Forged")
            else:
                st.error("Models are missing. Cannot analyze the signature without the trained models.")

if __name__ == "__main__":
    main()
