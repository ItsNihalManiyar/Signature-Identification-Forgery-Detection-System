# Project Abstract: College Signature Identification & Forgery Detection System

## Introduction
Biometric authentication has become an essential pillar of securing digital and physical assets in modern institutions. Among the various biometric modalities, handwriting and signature verification remain highly relevant, particularly in academic and financial environments where authorization relies on handwritten consent. The "College Signature Identification & Forgery Detection System" is a robust, automated solution designed to authenticate student identities and detect fraudulent signatures in real time. This project addresses the critical need for a reliable, non-intrusive biometric verification system by leveraging state-of-the-art computer vision and deep learning techniques to analyze signature images. By eliminating the reliance on subjective human inspection, this system ensures high accuracy, scalability, and security for administrative processes within educational institutions.

## Methodology and Dataset
The foundation of this system is the widely recognized CEDAR signature dataset, which comprises a comprehensive collection of genuine and forged signatures. For this project, the dataset includes samples from 55 distinct writers, with 24 genuine and 24 forged signatures per writer, forming a diverse and challenging dataset.

To prepare the raw images for neural network classification, an automated OpenCV-based image preprocessing pipeline was implemented. The preprocessing module systematically reads the valid image files and undergoes several critical transformations: first, it is converted to grayscale to reduce computational complexity by discarding extraneous color information. Next, Otsu's Thresholding algorithm is applied to effectively binarize the image, yielding a clean signature with prominent strokes. Finally, the images are uniformly resized to a standard 224x224 pixel dimension, ensuring compatibility with the input requirements of the subsequent deep learning models.

## Neural Network Architecture
At the core of the system is a dual-model architecture relying on Convolutional Neural Networks (CNNs). CNNs are remarkably well-suited for image-based tasks due to their ability to learn spatial hierarchies of features. 

1. **Identification Model**: The first CNN is an ID_Model designed for multi-class classification. It consists of three convolutional layers with ReLU activation to extract spatial features, grouped with MaxPooling2D layers to downsize the feature maps. The flattened output is passed through fully connected layers with Dropout for regularization. The final Dense layer uses a Softmax activation function to evaluate the signature and recognize the author across 55 assigned student names.
2. **Verification Model**: The second CNN is a Verify_Model structured similarly to extract high-level feature representations but is uniquely optimized for a binary objective. Its final layer uses a Sigmoid activation to compute a distinct probability score, distinguishing strictly whether the analyzed signature is a genuine sample or an unauthorized forgery.

## Application and Impact
For practical application, the predictive models are integrated into an interactive web application built with Streamlit. The dashboard features a clean interface where users can drag-and-drop or upload signature image files. Upon submission, the script triggers the OpenCV preprocessing step dynamically, feeds the normalized data to both Convolutional Neural Networks, and visually presents the precise identification outcome alongside an authoritative status badge denoting "Genuine" or "Forged". 

Ultimately, this project highlights the seamless integration of computer vision pipelines, complex dataset synthesis methods, and dual-model deep learning design to deliver an end-to-end software product capable of automating strict, verifiable biometric authentication for modern college systems.
