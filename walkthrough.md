# College Signature Identification & Forgery Detection System Walkthrough

All requirements for the College Signature Identification & Forgery Detection System have been successfully implemented. The following modules and files are now available in your project directory:

## 1. Environment & Setup
- **File:** [requirements.txt](file:///Users/nih5586/PyCharmMiscProject/ML_A8/requirements.txt)
- Contains all required dependencies such as `tensorflow`, `opencv-python`, `streamlit`, `pandas`, and `scikit-learn`.

## 2. Image Preprocessing
- **File:** [preprocess_images.py](file:///Users/nih5586/PyCharmMiscProject/ML_A8/preprocess_images.py)
- Utilizes OpenCV to read images from the raw CEDAR dataset (`full_org` and `full_forg`), converts them to grayscale, applies Otsu’s Thresholding to binarize the signatures accurately, and resizes them to 224x224. Outputs are saved seamlessly to `processed_data/`.

## 3. Dataset Creation & Mapping
- **File:** [create_dataset.py](file:///Users/nih5586/PyCharmMiscProject/ML_A8/create_dataset.py)
- Maps the 55 student IDs to 55 common Indian student names. It scans the `processed_data/` directory, extracts the writer IDs from the filenames, assigns labels (0 for Genuine, 1 for Forged), and compiles everything into a unified Pandas DataFrame saved as `dataset.csv`.

## 4. Dual-Model Training Architecture
- **File:** [train_models.py](file:///Users/nih5586/PyCharmMiscProject/ML_A8/train_models.py)
- Implements two Convolutional Neural Networks using TensorFlow/Keras:
  - **ID_Model:** A CNN ending with a 55-node Softmax layer to predict the exact student ID.
  - **Verify_Model:** A Binary CNN ending with a Sigmoid layer to detect whether a signature is genuine or forged.
- The models correctly load `dataset.csv`, normalize the images, and perform training.

## 5. User Interface
- **File:** [app.py](file:///Users/nih5586/PyCharmMiscProject/ML_A8/app.py)
- Provides an elegant Streamlit Dashboard allowing users to upload a signature. It performs real-time preprocessing, queries the ID, maps it to the generated student names, and visually confirms the authentication status (Genuine in green, Forged in red).

## 6. Project Abstract
- **File:** [abstract.md](file:///Users/nih5586/PyCharmMiscProject/ML_A8/abstract.md)
- A professionally written 500-word documentation detailing the system architecture, preprocessing methodology with OpenCV, Convolutional Neural Networks, and the biometric verification use case.

## Next Steps / Execution Instructions:
Open a terminal in the `/Users/nih5586/PyCharmMiscProject/ML_A8/` directory and run:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Preprocessing & Data Mapping:**
   ```bash
   python preprocess_images.py
   python create_dataset.py
   ```
3. **Train Models:**
   ```bash
   python train_models.py
   ```
4. **Launch Streamlit App:**
   ```bash
   streamlit run app.py
   ```
