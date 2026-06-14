# College Signature Identification & Forgery Detection System

A professional-grade biometric authentication system developed using the CEDAR dataset, OpenCV, and TensorFlow/Keras.

## Features
- **Dual CNN Architecture:** One model for student identification (55 classes) and one for binary forgery detection.
- **Advanced Preprocessing:** Automated pipeline using Otsu's Thresholding and grayscale normalization.
- **Interactive Dashboards:** Includes a standard (`app.py`) and an advanced (`advanced_app.py`) Streamlit UI.
- **Real-time Feedback:** Visual telemetry for image processing steps.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess & Map Data:**
   ```bash
   python preprocess_images.py
   python create_dataset.py
   ```
3. **Train Models:**
   ```bash
   python train_models.py
   ```
4. **Run Application:**
   ```bash
   streamlit run advanced_app.py
   ```

## Repository Structure
- `preprocess_images.py`: OpenCV preprocessing script.
- `create_dataset.py`: ID-to-Name mapping and CSV generator.
- `train_models.py`: Dual-model CNN training logic.
- `app.py`: Standard Streamlit dashboard.
- `advanced_app.py`: Feature-rich Streamlit dashboard.
- `abstract.md`: Detailed technical project abstract.
- `walkthrough.md`: Project guide and GitHub instructions.

## Screenshorts 

<img width="1470" height="956" alt="" src="https://github.com/user-attachments/assets/b63eb89f-e555-46cb-b812-64a1c7825960" />

